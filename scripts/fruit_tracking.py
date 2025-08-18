#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, Pose, PoseStamped, PoseArray, PoseWithCovarianceStamped
from aoc_fruit_detector.msg import FruitInfoMessage, FruitInfoArray
from visualization_msgs.msg import Marker, MarkerArray
import argparse
import random

import os, yaml, cv2
from detectron_predictor.detectron_predictor import DetectronPredictor

from cv_bridge import CvBridge, CvBridgeError

from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np

import image_geometry

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from detectron_predictor.json_writer.pycococreator.pycococreatortools.fruit_orientation import FruitTypes

import matplotlib.pyplot as plt

import tf2_ros
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as R

from scipy.stats import zscore

class FruitTrackingNode(Node):
    def __init__(self, non_ros_config_path):
        super().__init__('aoc_fruit_tracker')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('min_depth', 0.1),
                ('max_depth', 15.0),
                ('constant_depth_value', 1.0),
                ('fruit_type', "strawberry"),
                ('pose3d_frame', ''),
                ('pose3d_tf', False),
                ('verbose', [False, False, False, True, True]),
                ('pub_verbose', False),
                ('pub_markers', False),
                ('use_ros', True)
            ]
        )

        self.package_name = 'aoc_fruit_detector'
        if non_ros_config_path:
            with open(non_ros_config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                
                for section in ['files', 'directories']:
                    if section in config_data:
                        for key, path in config_data[section].items():
                            if path.startswith('./'):
                                package_share_directory = get_package_share_directory(self.package_name)
                                config_data[section][key] = os.path.join(package_share_directory, path.lstrip('./'))
                
                self.image_dir = config_data['directories']['test_image_dir']
                self.prediction_json_dir = config_data['directories']['prediction_json_dir']
                self.prediction_output_dir = config_data['directories']['prediction_output_dir']

                self.det_predictor = DetectronPredictor(config_data)
                fruit_type = config_data['settings']['fruit_type']
                if (fruit_type.upper()=="STRAWBERRY"):
                    self.fruit_type=FruitTypes.Strawberry
                elif (fruit_type.upper()=="TOMATO"):
                    self.fruit_type=FruitTypes.Tomato
                else:
                    self.fruit_type=FruitTypes.Strawberry
        else:
            raise FileNotFoundError(f"No config file found in any ' {self.package_name}/config/' folder within {os.getcwd()}")

        self.use_ros = self.get_parameter('use_ros').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.max_depth = 1.0

        self.threshold = 0.7

        self.marker_array = MarkerArray()

        if self.use_ros:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            # Get parameter values from ROS2 network
            self.constant_depth_value = self.get_parameter('constant_depth_value').value
            if self.get_parameter('fruit_type').value == "tomato":
                self.tomato = True
            elif self.get_parameter('fruit_type').value == "strawberry":
                self.tomato = False
            self.pose3d_frame = self.get_parameter('pose3d_frame').value

            self.pose3d_tf = self.get_parameter('pose3d_tf').value

            self.draw_centroid = self.get_parameter('verbose').value[0]
            self.draw_bbox = self.get_parameter('verbose').value[1]
            self.draw_mask = self.get_parameter('verbose').value[2]
            self.draw_cf = self.get_parameter('verbose').value[3]
            self.add_text = self.get_parameter('verbose').value[4]

            self.pub_verbose = self.get_parameter('pub_verbose').value
            self.pub_markers = self.get_parameter('pub_markers').value

            self.bridge = CvBridge()
            self.camera_model = image_geometry.PinholeCameraModel()
            self.set_default_camera_model() # in case camera_info calibration message not available 
            
            # Declare subscribers
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=1
            )

            self.image_sub = self.create_subscription(
                Image,
                'camera/image_raw',
                self.image_callback,
                qos_profile
            )

            self.depth_sub = self.create_subscription(
                Image,
                'camera/depth',
                self.depth_callback,
                qos_profile
            )

            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                'camera/camera_info',
                self.camera_info_callback,
                qos_profile
            )

            # Create and declare publishers
            self.publisher_comp = self.create_publisher(Image, 'image_composed', 5)
            self.publisher_3dmarkers = self.create_publisher(MarkerArray, 'fruit_markers', 5)
            self.publisher_fruit_pose = self.create_publisher(PoseWithCovarianceStamped, 'fruit_pose_3d', 5)
            self.publisher_fruit_pose_left = self.create_publisher(PoseWithCovarianceStamped, 'fruit_pose_3d_left', 5)
            self.publisher_fruit_pose_right = self.create_publisher(PoseWithCovarianceStamped, 'fruit_pose_3d_right', 5)
        else: 
            all_files = sorted([f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))])

            rgb_files = sorted([f for f in all_files if 'image' in f])
            depth_files = sorted([f for f in all_files if 'depth' in f])

            sample_no = 1
            for rgb_file in rgb_files:
                corr_depth_file = rgb_file.replace('image', 'depth', 1)

                if corr_depth_file in depth_files:
                    image_file_name=os.path.join(self.image_dir, rgb_file)
                    depth_file_name = os.path.join(self.image_dir, corr_depth_file)
                    rgb_image = cv2.imread(image_file_name)  # bgr8
                    depth_image = cv2.imread(depth_file_name, cv2.IMREAD_UNCHANGED)
                    depth_image = np.nan_to_num(depth_image, nan=self.max_depth, posinf=self.max_depth, neginf=self.min_depth)
                    rgbd_image = np.dstack((rgb_image, depth_image))
                    filename, extension = os.path.splitext(rgb_file)
                    if (self.prediction_json_dir!=""):
                        os.makedirs(self.prediction_json_dir, exist_ok=True)
                        prediction_json_output_file = os.path.join(self.prediction_json_dir, filename)+'.json'
                    self.det_predictor.get_predictions_image(rgbd_image, prediction_json_output_file, self.prediction_output_dir, image_file_name, sample_no, self.fruit_type)
                else:
                    self.get_logger().warn(f"Warning: No corresponding depth file: {corr_depth_file} for rgb file: {rgb_file}")
                sample_no += 1
    
    def publish_fruit_markers(self, fruit_dict):
        delete_marker_array = MarkerArray()

        for marker in self.marker_array.markers:
            delete_marker = Marker()
            delete_marker = marker
            delete_marker.action = Marker.DELETE
            delete_marker_array.markers.append(delete_marker)
        self.publisher_3dmarkers.publish(delete_marker_array)
        
        marker_array = MarkerArray()
        # Loop over the detected fruits and create a markers for each one
        for i, fruit_data in fruit_dict.items():
            if float(fruit_data.get('confidence')) > self.threshold:
                markers = self.create_fruit_marker(fruit_data, i)  # Create markers with unique ID: i*10000 + marker_cnt
                for marker in markers.markers:
                    marker_array.markers.append(marker)

        # Publish the marker array
        self.publisher_3dmarkers.publish(marker_array)
        self.marker_array = marker_array

    def create_fruit_marker(self, fruit_data, marker_id):
        markers = MarkerArray()

        pose3d = fruit_data.get('pose3d')
        pose_array = fruit_data.get('pose_array')
        pose_centroid = fruit_data.get('pose_centroid')
        pose_left = fruit_data.get('pose_left')
        pose_right = fruit_data.get('pose_right')

        color_r = random.uniform(0.0, 1.0)
        color_g = random.uniform(0.0, 1.0)
        color_b = random.uniform(0.0, 1.0)

        if pose_centroid and len(pose_array.poses) > 0:
            marker = Marker()
            marker.header.frame_id = self.pose3d_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fruit_centroids"
            marker.id = marker_id
            marker.type = Marker.SPHERE 
            marker.action = Marker.ADD
            marker.pose = pose_centroid
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.r = color_r
            marker.color.g = color_g
            marker.color.b = color_b
            marker.color.a = 1.0
            marker.lifetime = rclpy.time.Duration(seconds=0).to_msg()

            markers.markers.append(marker)

        if pose_left and len(pose_array.poses) > 0:
            marker = Marker()
            marker.header.frame_id = self.pose3d_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fruit_left_bound"
            marker.id = marker_id+2000
            marker.type = Marker.SPHERE 
            marker.action = Marker.ADD
            marker.pose = pose_left
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.r = color_r
            marker.color.g = color_g
            marker.color.b = color_b
            marker.color.a = 1.0
            marker.lifetime = rclpy.time.Duration(seconds=0).to_msg()

            markers.markers.append(marker)

        if pose_right and len(pose_array.poses) > 0:
            marker = Marker()
            marker.header.frame_id = self.pose3d_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fruit_right_bound"
            marker.id = marker_id+1000
            marker.type = Marker.SPHERE 
            marker.action = Marker.ADD
            marker.pose = pose_right
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.r = color_r
            marker.color.g = color_g
            marker.color.b = color_b
            marker.color.a = 1.0
            marker.lifetime = rclpy.time.Duration(seconds=0).to_msg()

            markers.markers.append(marker)

        if pose_array and len(pose_array.poses) > 0:
            for i, pose in enumerate(pose_array.poses):
                marker = Marker()

                marker.header.frame_id = self.pose3d_frame
                marker.header.stamp = self.get_clock().now().to_msg()

                marker.ns = "fruits"
                marker.id = marker_id*10000 + i

                marker.type = Marker.SPHERE 
                marker.action = Marker.ADD
                marker.pose = pose

                # Set the scale of the marker (size of the fruit marker)
                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02

                marker.color.r = color_r
                marker.color.g = color_g
                marker.color.b = color_b
                marker.color.a = 1.0

                marker.lifetime = rclpy.time.Duration(seconds=0).to_msg()

                markers.markers.append(marker)
        elif pose3d:
            marker = Marker()
            marker.header.frame_id = self.pose3d_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fruits"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose = pose3d.pose

            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02

            marker.color.r = color_r
            marker.color.g = color_g
            marker.color.b = color_b
            marker.color.a = 1.0

            marker.lifetime = rclpy.time.Duration(seconds=0).to_msg()
            markers.markers.append(marker)
        
        return markers

    def compute_pose_array(self, segmentation, depth_image, centroid_2d):
        pose_array = PoseArray()
        segmentation_coordinates = np.array(segmentation).reshape(-1, 2).astype(np.int32)
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [segmentation_coordinates], 1)
        y_coords, x_coords = np.where(mask == 1)

        # segmentation_coordinates = np.array(segmentation).reshape(-1, 2)
        # x_coords, y_coords = segmentation_coordinates[:, 0].astype(int), segmentation_coordinates[:, 1].astype(int)
        # valid_indices = (y_coords >= 0) & (y_coords < depth_image.shape[0]) & (x_coords >= 0) & (x_coords < depth_image.shape[1])
        # x_coords, y_coords = x_coords[valid_indices], y_coords[valid_indices]
        
        depth_values = []
        for x, y in zip(x_coords, y_coords):
            if self.min_depth < depth_image[y, x] < self.max_depth:
                depth_values.append([x, y, depth_image[y, x]])

        # Convert to a NumPy array
        depth_values_array = np.array(depth_values)  # Shape (N, 3) with [x, y, depth]
        if depth_values_array.size == 0:
            self.get_logger().warn("No valid depth values found within the specified mask.")
            return PoseArray(), Pose()
        
        depth_column = depth_values_array[:, 2]

        # filtering using Median Absolute Deviation (MAD)
        median_depth = np.median(depth_column)
        mad = np.median(np.abs(depth_column - median_depth))
        mad_threshold = 3

        if mad != 0:  # to avoid division by zero
            mad_scores = np.abs(depth_column - median_depth) / (1.4826 * mad)
            filtered_depth_values = depth_values_array[mad_scores < mad_threshold]
        else:
            filtered_depth_values = depth_values_array

        x_c, y_c = centroid_2d # Centroid XY
        closest_pose3d = None
        min_distance = float('inf')

        for x, y, depth in filtered_depth_values:
            pose3d = Pose()
            ray = self.back_project_2d_to_3d_ray(x, y)
            p_3d_camera_frame = self.compute_3d_point_from_depth(ray, depth)
            pose3d.position.x = p_3d_camera_frame[0]
            pose3d.position.y = p_3d_camera_frame[1]
            pose3d.position.z = p_3d_camera_frame[2]
            pose3d.orientation.x = 0.0
            pose3d.orientation.y = 0.0
            pose3d.orientation.z = 0.0
            pose3d.orientation.w = 1.0
            pose_array.poses.append(pose3d)
            if (x == x_c) and (y == y_c):
                pose_centroid = pose3d
            else:
                distance = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_pose3d = pose3d

        if 'pose_centroid' not in locals():
            pose_centroid = closest_pose3d

        pose_array.header.frame_id = self.pose3d_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()

        cov_xx = 0.00022*0.00022  # std_x^2
        cov_yy = 0.00022*0.00022  # std_y^2
        cov_zz = 0.00124*0.00124  # std_z^2

        # Orientation variances (high since not measured)
        h_var = 1e6

        # Covariance matrix (6x6 flattened)
        covariance_matrix = [
            cov_xx, 0,      0,      0,      0,      0,   # Row 1
            0,      cov_yy, 0,      0,      0,      0,   # Row 2
            0,      0,      cov_zz, 0,      0,      0,   # Row 3
            0,      0,      0,      h_var, 0, 0, # Row 4
            0,      0,      0,      0, h_var, 0, # Row 5
            0,      0,      0,      0,      0, h_var  # Row 6
        ]

        pose_c = PoseWithCovarianceStamped()
        pose_c.header.frame_id = self.pose3d_frame
        pose_c.header.stamp = pose_array.header.stamp
        pose_c.pose.pose = pose_centroid
        pose_c.pose.covariance = covariance_matrix
        self.publisher_fruit_pose.publish(pose_c)

        pose_left = Pose()
        pose_right = Pose()

        if len(pose_array.poses) > 0:
            x_values = [pose.position.x for pose in pose_array.poses]
            diameter = max(x_values) - min(x_values)
            radius = diameter / 2

            x_center = pose_centroid.position.x
            z_center = pose_centroid.position.z + radius

            x_left = x_center - radius
            x_right = x_center + radius
  
            pose_left.position.y = pose_centroid.position.y  
            pose_left.position.x = x_left
            pose_left.position.z = z_center

            pose_right.position.y = pose_centroid.position.y   
            pose_right.position.x = x_right
            pose_right.position.z = z_center
            
            # Covariance matrix (6x6 flattened)
            covariance_matrix = [
                100*cov_xx, 0,      0,          0,      0,      0,      # Row 1
                0,          cov_yy, 0,          0,      0,      0,      # Row 2
                0,          0,      25*cov_zz,  0,      0,      0,      # Row 3
                0,          0,      0,          h_var,  0,      0,      # Row 4
                0,          0,      0,          0,      h_var,  0,      # Row 5
                0,          0,      0,          0,      0,      h_var   # Row 6
            ]

            pose_l = PoseWithCovarianceStamped()
            pose_l.header.frame_id = self.pose3d_frame
            pose_l.header.stamp = pose_array.header.stamp
            pose_l.pose.pose = pose_left
            pose_l.pose.covariance = covariance_matrix

            pose_r = PoseWithCovarianceStamped()
            pose_r.header.frame_id = self.pose3d_frame
            pose_r.header.stamp = pose_array.header.stamp
            pose_r.pose.pose = pose_right
            pose_r.pose.covariance = covariance_matrix

            self.publisher_fruit_pose_left.publish(pose_l)
            self.publisher_fruit_pose_right.publish(pose_r)
        
        return pose_array, pose_centroid, pose_left, pose_right

    def compute_pose3d(self, fruit_dict, fruit_id, depth_image):
        pose3d = PoseStamped()

        height, width = depth_image.shape
        centroid_2d = fruit_dict.get(fruit_id, {}).get('centroid_2d', ['0.0','0.0'])
        x = int(centroid_2d[0])  # Centroid X
        y = int(centroid_2d[1])  # Centroid Y

        if 0 <= x < width and 0 <= y < height:
            depth_values_at_pose = depth_image[y, x]
            non_zero_depth_values = depth_values_at_pose[depth_values_at_pose > 0]

            if non_zero_depth_values.size > 0:
                closest_depth_value = np.min(non_zero_depth_values)
            else:
                closest_depth_value = self.max_depth 
        else:
            closest_depth_value = self.max_depth
            self.get_logger().warn(f'Out of size x:{x}, width:{width}, y:{y} and height:{height}')
        
        ray = self.back_project_2d_to_3d_ray(x, y)
        p_3d_camera_frame = self.compute_3d_point_from_depth(ray, closest_depth_value)
        #self.get_logger().info(f'3D point at depth {closest_depth_value}: [{p_3d_camera_frame[0]:.2f}, {p_3d_camera_frame[1]:.2f}, {p_3d_camera_frame[2]:.2f}]')

        pose3d.pose.position.x = p_3d_camera_frame[0]
        pose3d.pose.position.y = p_3d_camera_frame[1]
        pose3d.pose.position.z = p_3d_camera_frame[2]
        
        # Identity quaternion (no rotation)
        pose3d.pose.orientation.x = 0.0
        pose3d.pose.orientation.y = 0.0
        pose3d.pose.orientation.z = 0.0
        pose3d.pose.orientation.w = 1.0  # No rotation

        pose3d.header.frame_id = self.pose3d_frame
        pose3d.header.stamp = self.get_clock().now().to_msg()
    
        return pose3d

    def set_default_camera_model(self):
        """
        Sets the camera model to default intrinsic parameters (pinhole model)
        """
        default_fx = 699.05  # Focal length in x (pixels)
        default_fy = 699.05  # Focal length in y (pixels)
        default_cx = 639.74  # Principal point x (image center in pixels)
        default_cy = 374.974  # Principal point y (image center in pixels)
        image_width = 1280
        image_height = 720

        # Create a fake CameraInfo message to initialize the camera model
        camera_info = CameraInfo()
        camera_info.width = image_width
        camera_info.height = image_height
        camera_info.distortion_model = "plumb_bob"  # Default distortion model

        # Set the intrinsic camera matrix K (3x3 matrix)
        camera_info.k = [default_fx, 0.0, default_cx, 0.0, default_fy, default_cy, 0.0, 0.0, 1.0]

        # Set the projection matrix P (3x4 matrix)
        camera_info.p = [default_fx, 0.0, default_cx, 0.0, 0.0, default_fy, default_cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        # Set default distortion coefficients (D)
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Initialize the camera model with the default CameraInfo
        self.camera_model.fromCameraInfo(camera_info)

        self.get_logger().info("Default camera model initialised with intrinsic parameters")

    def camera_info_callback(self, msg):
        
        self.from_camera_info(msg)
        #self.camera_model, self.distortion_coeffs = self.from_camera_info(msg)

        self.pose3d_frame = msg.header.frame_id

        #self.get_logger().info('Camera model acquired from camera_info message and initialised with intrinsic parameters')
    
    def from_camera_info(self, msg):
        self.camera_model.fromCameraInfo(msg)
        #camera_matrix = np.array(msg.k).reshape(3, 3)
        #distortion_coeffs = np.array(msg.d)
        #return camera_matrix, distortion_coeffs

    def back_project_2d_to_3d_ray(self, u, v):
        ray = self.camera_model.projectPixelTo3dRay((u, v))
        return ray
        #pixel = np.array([[u, v]], dtype=np.float32)
        #pixel = np.expand_dims(pixel, axis=0)
        #undistorted_point = cv2.undistortPoints(pixel, self.camera_model, self.distortion_coeffs)
        #ray = [undistorted_point[0][0][0], undistorted_point[0][0][1], 1.0]
        
        #return ray

    def compute_3d_point_from_depth(self, ray, depth):
        # Compute the 3D point in optical frame
        point_optical = np.array([ray[0] * depth, ray[1] * depth, ray[2] * depth, 1.0])
        if self.pose3d_tf:
            # Apply the transformation to convert to camera frame
            point_camera = np.dot(self.tf_matrix, point_optical)
            return point_camera[:3]
        else:
            return point_optical
        

    def depth_callback(self, msg):
        try:
            # Convert ROS2 depth Image message to OpenCV depth image
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            # To replace NaN and Inf with max_depth value
            self.cv_depth_image[np.isnan(self.cv_depth_image)] = self.max_depth
            self.cv_depth_image[np.isinf(self.cv_depth_image)] = self.max_depth
            self.depth_msg = msg 
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
    
    def create_fruit_dict(self, pose_list, confidence_list, confidence_threshold=0.5):
        confidence_dict = {entry['annotation_id']: entry['confidence'] for entry in confidence_list}
        
        return {
            entry['annotation_id']: {
                'centroid_2d': entry['centroid'],
                'orientation': entry['orientation'],
                'confidence': confidence_dict.get(entry['annotation_id'], '-1.0')
            }
            for entry in pose_list
            if float(confidence_dict.get(entry['annotation_id'], '-1.0')) >= self.threshold
        }     

    def get_optic_tf(self):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'zed_camera_link', 'zed_left_camera_optical_frame', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5)
            )
            tr = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            q = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, TimeoutError):
            tr = [0.0, 0.0, 0.0]
            q = R.from_quat([0.5, -0.5, 0.5, -0.5]).as_quat() # default orientation between camera and optical frames
            self.get_logger().warn(f"Default transform between camera and optical frame used: t={tr} and q={q}")
        rot_matrix = R.from_quat(q).as_matrix()
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = rot_matrix
        #print(f"rot_matrix: {rot_matrix}")
        tf_matrix[:3, 3] = tr
        #print(f"tf_matrix: {tf_matrix}")
        return tf_matrix

    def image_callback(self, msg):
        try:
            self.get_logger().info("Image captured.")

            self.tf_matrix = self.get_optic_tf()

            # Convert ROS Image message to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info("RGB ready.")
            
            # Create image_id as an integer using the timestamp
            image_id = int(f'{msg.header.stamp.sec}{str(msg.header.stamp.nanosec).zfill(9)}')
            #self.get_logger().info(f"Image ID is: {image_id}")

            if hasattr(self, 'cv_depth_image') and self.cv_depth_image is not None:
                # Ensure that the depth image is the same size as the RGB image
                if self.cv_image.shape[:2] != self.cv_depth_image.shape[:2]:
                    self.get_logger().warn("Resizing depth image to match RGB image dimensions.")
                    depth_image = cv2.resize(self.cv_depth_image, (self.cv_image.shape[1], self.cv_image.shape[0]))
                else:
                    depth_image = self.cv_depth_image
            else:
                # If no depth image is available, use the constant depth value
                self.get_logger().warn(f"No depth image available. Using constant depth value: {self.constant_depth_value}")
                depth_image = np.full(self.cv_image.shape[:2], self.constant_depth_value, dtype=np.float32)

            self.get_logger().info("Depth ready")
            # Combine RGB and depth into a single 4-channel image (3 for RGB + 1 for depth)
            rgbd_image = np.dstack((self.cv_image, depth_image))
            self.get_logger().info("RGBD ready")
            
            json_annotation_message = self.det_predictor.get_predictions_message_short(rgbd_image, image_id, self.fruit_type)

            #merged_mask = np.bitwise_or(rgb_masks[:, :, 0], rgb_masks[:, :, 1])

            annotations = json_annotation_message.get('annotations', [])
            confidence_list = json_annotation_message.get('confidence', [])
            pose_list = json_annotation_message.get('orientation', [])
            
            fruit_dict = self.create_fruit_dict(pose_list, confidence_list)

            for annotation in annotations:
                #self.get_logger().info(f'Annotation: {annotation}')
                fruit_id = annotation.get('id', None)
                if float(fruit_dict.get(fruit_id, {}).get('confidence', '-1.0')) >= self.threshold:
                    area = float(annotation.get('area', 0.0))
                    segmentation = annotation.get('segmentation', [])
                    segmentation = [point for sublist in segmentation for point in sublist]  # Flatten segmentation
                    
                    centroid_2d = fruit_dict.get(fruit_id, {}).get('centroid_2d', ['0.0','0.0'])
                    centroid_2d = [int(float(coord)) for coord in centroid_2d]

                    #self.get_logger().info(f"fruit_id: {fruit_id}")
                    #self.get_logger().info(f"centroid_2d: {centroid_2d}")
                    #self.get_logger().info(f"segmentation: {segmentation}")
                    pose_array, pose_centroid, pose_left, pose_right = self.compute_pose_array(segmentation, depth_image, centroid_2d)
                    if fruit_id in fruit_dict:
                        fruit_dict[fruit_id].update({'segmentation': segmentation, 'area': area, 'pose_array': pose_array, 'pose_centroid': pose_centroid, 'pose_left': pose_left, 'pose_right': pose_right})

            largest_fruit_id = max(fruit_dict, key=lambda fruit_id: float(fruit_dict[fruit_id]['area']))
            safest_fruit_id = max(fruit_dict, key=lambda fruit_id: float(fruit_dict[fruit_id]['confidence']))

            if largest_fruit_id == safest_fruit_id:
                fruit_dict = {largest_fruit_id: fruit_dict[largest_fruit_id]}

            self.get_logger().info("Annotations ready")

            if self.pub_verbose:
                rgb_image_composed = self.add_markers_on_image(self.cv_image, fruit_dict)
                self.publisher_comp.publish(rgb_image_composed)
            if self.pub_markers:
                self.publish_fruit_markers(fruit_dict)
            self.get_logger().info("Published")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def add_markers_on_image(self, cv_image, fruit_dict):
        height, width, _ = cv_image.shape  # Get image dimensions
        scale_factor = min(width, height) / 1500  # Scale the circle size based on image dimensions
        
        for fruit_id, data in fruit_dict.items():
            if float(data.get('confidence')) > self.threshold:
                centroid = data.get('centroid_2d')
                x = int(centroid[0])
                y = int(centroid[1])
                theta = np.deg2rad(float(data.get('orientation')))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        
                if self.draw_centroid:
                    # Point the centroid (origin of the fruit)
                    radius = int(10 * scale_factor)
                    cv2.circle(cv_image, (x, y), radius, color, -1)
                
                if self.draw_mask:
                    # Draw the polygon mask outline
                    segmentation = data.get('segmentation', [])  # Retrieve segmentation from data
                    mask_points = np.array(segmentation, dtype=np.int32).reshape((-1, 2))  # Convert 1D mask to Nx2 format
                    cv2.polylines(cv_image, [mask_points], isClosed=True, color=color, thickness=2)

                if self.draw_cf:
                    # Draw fruit coordinate frame (cf)
                    color_x = (0, 0, 255) # Red for x axis
                    color_y = (0, 255, 0)  # Green for y axis
                    arrow_length = int(30 * scale_factor)
                    end_x_x = int(x + arrow_length * np.cos(theta))  # Calculate endpoint x
                    end_y_x = int(y + arrow_length * np.sin(theta))  # Calculate endpoint y
                    cv2.arrowedLine(cv_image, (x, y), (end_x_x, end_y_x), color_x, thickness=2, tipLength=0.3)
                    end_x_y = int(x + arrow_length * np.cos(theta+np.pi/2))  # Calculate endpoint x
                    end_y_y = int(y + arrow_length * np.sin(theta+np.pi/2))  # Calculate endpoint y
                    cv2.arrowedLine(cv_image, (x, y), (end_x_y, end_y_y), color_y, thickness=2, tipLength=0.3)

                if self.add_text:
                    text_position = (end_x_x + 5, end_y_x - 5)  # Offset text slightly from the arrow tip
                    font_scale = 1.5 * scale_factor  # Adjust text size based on image size
                    font_thickness = max(1, int(2 * scale_factor))  # Scale text thickness
                    cv2.putText(
                        cv_image, 
                        f"{float(data.get('orientation')):.1f}",  # Format theta to 1 decimal place
                        text_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        color,
                        thickness=font_thickness, 
                        lineType=cv2.LINE_AA
                    )

        # Convert the modified image back to a ROS image message
        composed_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        return composed_image

def main(args=None):
    parser = argparse.ArgumentParser(description='Fruit Tracking Node')
    parser.add_argument('--config-file', required=True, help='Path to non-ROS parameters YAML file')
    non_ros_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = FruitTrackingNode(non_ros_config_path=non_ros_args.config_file)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()