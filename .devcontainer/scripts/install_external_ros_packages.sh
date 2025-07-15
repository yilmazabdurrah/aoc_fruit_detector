#!/bin/bash

# Exit immediately if a command exits with a non-zero status and enable debug mode for verbose output
set -e -x

# Update rosdep database to ensure all dependencies are up-to-date
rosdep update
sudo apt-get update 

# Navigate to the src directory of the current workspace
WS=$1
echo "WS dir ${WS}"
mkdir -p ${WS}/src/external_packages 
cd ${WS}/src/external_packages 

# Import repositories using vcs
echo "Importing repositories with vcs..."
vcs import < ${WS}/src/repos/external.repos 

# Install any dependencies specified in the cloned repositories' package.xml files
# The '-r' flag makes the process recursive
# The '-y' flag automatically answers "yes" to any prompts
rosdep install -r -y -i --from-paths .

# Install PyTorch in user context to ensure it's available for detectron2 setup
# Using the same versions and CUDA support as configured in the Docker build
python3 -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

# Install detectron2 in editable mode
python3 -m pip install -e detectron2

echo "Setup complete."
