#!bin/bash

sudo apt install cmake libncurses5-dev libncursesw5-dev git
git clone https://github.com/Syllo/nvtop.git
mkdir -p nvtop/build && cd nvtop/build
cmake ..

# If an error is reported:"Could NOT find NVML (missing: NVML_INCLUDE_DIRS)"
# Then please use the following command
# cmake ..-DNVML_RETRIEVE_HEADER_ONLINE=True

make
sudo make install #Need root privileges, if you report a privilege error, please add sudo