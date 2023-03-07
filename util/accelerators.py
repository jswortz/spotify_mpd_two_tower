# Single machine, single GPU, 80 GB 'NVIDIA_A100_80GB'
# WORKER_MACHINE_TYPE = 'a2-ultragpu-1g' # 80 GB
# REPLICA_COUNT = 1
# ACCELERATOR_TYPE = 'NVIDIA_A100_80GB'
# PER_MACHINE_ACCELERATOR_COUNT = 1
# REDUCTION_SERVER_COUNT = 0                                                      
# REDUCTION_SERVER_MACHINE_TYPE = "n1-highcpu-16"
# DISTRIBUTE_STRATEGY = 'single'

# # # Single Machine; multiple GPU
# WORKER_MACHINE_TYPE = 'a2-highgpu-4g' # a2-ultragpu-4g
# REPLICA_COUNT = 1
# ACCELERATOR_TYPE = 'NVIDIA_TESLA_A100'
# PER_MACHINE_ACCELERATOR_COUNT = 4
# REDUCTION_SERVER_COUNT = 0                                                      
# REDUCTION_SERVER_MACHINE_TYPE = "n1-highcpu-16"
# DISTRIBUTE_STRATEGY = 'mirrored'

# # # # Multiple Machine; 1 GPU per machine
# WORKER_MACHINE_TYPE = 'a2-highgpu-2g' # a2-ultragpu-4g
# REPLICA_COUNT = 2
# ACCELERATOR_TYPE = 'NVIDIA_TESLA_A100'
# PER_MACHINE_ACCELERATOR_COUNT = 2
# REDUCTION_SERVER_COUNT = 4                                                      
# REDUCTION_SERVER_MACHINE_TYPE = "n1-highcpu-16"
# DISTRIBUTE_STRATEGY = 'multiworker'

# # # Multiple Machines, 1 GPU per Machine
# WORKER_MACHINE_TYPE = 'n1-standard-16'
# REPLICA_COUNT = 9
# ACCELERATOR_TYPE = 'NVIDIA_TESLA_T4'
# PER_MACHINE_ACCELERATOR_COUNT = 1
# REDUCTION_SERVER_COUNT = 10                                                      
# REDUCTION_SERVER_MACHINE_TYPE = "n1-highcpu-16"
# DISTRIBUTE_STRATEGY = 'multiworker'