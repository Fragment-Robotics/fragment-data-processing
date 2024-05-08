FROM ros:humble-desktop

# Install addiional dependencies for mcap
RUN apt update && apt-get install -y \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-mcap    

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add sourcing to bashrc
RUN echo "source $HOME/.cargo/env" >> $HOME/.bashrc