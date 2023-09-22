# ===================================================
# get accelerator_config
# ===================================================
def get_accelerator_config(
    key: str = None, 
    accelerator_per_machine: int = 1, 
    worker_n: int = 1, 
    reduction_n: int = 0,
    reduction_machine_type: str = "n1-highcpu-16", 
    distribute: str = 'single'
):
    """
    returns GPU configuration for vertex training
    
    example:
        desired_config = get_accelerator_config(MY_CHOICE)
    """
    if key == "a100":
        WORKER_MACHINE_TYPE = 'a2-highgpu-1g'
        REPLICA_COUNT = worker_n
        ACCELERATOR_TYPE = 'NVIDIA_TESLA_A100'
        PER_MACHINE_ACCELERATOR_COUNT = accelerator_per_machine
        REDUCTION_SERVER_COUNT = reduction_n                                                 
        REDUCTION_SERVER_MACHINE_TYPE = reduction_machine_type
        DISTRIBUTE_STRATEGY = distribute
    elif key == 't4':
        WORKER_MACHINE_TYPE = 'n1-standard-16'
        REPLICA_COUNT = worker_n
        ACCELERATOR_TYPE = 'NVIDIA_TESLA_T4' # NVIDIA_TESLA_T4 NVIDIA_TESLA_V100
        PER_MACHINE_ACCELERATOR_COUNT = accelerator_per_machine
        DISTRIBUTE_STRATEGY = distribute
        REDUCTION_SERVER_COUNT = reduction_n                                                   
        REDUCTION_SERVER_MACHINE_TYPE = reduction_machine_type
    elif key == "False":
        WORKER_MACHINE_TYPE = 'n2-highmem-32' # 'n1-highmem-96'n | 'n2-highmem-92'
        REPLICA_COUNT = worker_n
        ACCELERATOR_TYPE = None
        PER_MACHINE_ACCELERATOR_COUNT = accelerator_per_machine
        DISTRIBUTE_STRATEGY = distribute
        REDUCTION_SERVER_COUNT = reduction_n                                                 
        REDUCTION_SERVER_MACHINE_TYPE = reduction_machine_type

    print(f"WORKER_MACHINE_TYPE            : {WORKER_MACHINE_TYPE}")
    print(f"REPLICA_COUNT                  : {REPLICA_COUNT}")
    print(f"ACCELERATOR_TYPE               : {ACCELERATOR_TYPE}")
    print(f"PER_MACHINE_ACCELERATOR_COUNT  : {PER_MACHINE_ACCELERATOR_COUNT}")
    print(f"DISTRIBUTE_STRATEGY            : {DISTRIBUTE_STRATEGY}")
    print(f"REDUCTION_SERVER_COUNT         : {REDUCTION_SERVER_COUNT}")
    print(f"REDUCTION_SERVER_MACHINE_TYPE  : {REDUCTION_SERVER_MACHINE_TYPE}")
    
    accelerator_dict = {
        "WORKER_MACHINE_TYPE": WORKER_MACHINE_TYPE,
        "REPLICA_COUNT": REPLICA_COUNT,
        "ACCELERATOR_TYPE": ACCELERATOR_TYPE,
        "PER_MACHINE_ACCELERATOR_COUNT": PER_MACHINE_ACCELERATOR_COUNT,
        "REDUCTION_SERVER_MACHINE_TYPE": REDUCTION_SERVER_MACHINE_TYPE,
        "DISTRIBUTE_STRATEGY": DISTRIBUTE_STRATEGY,
    }
    return accelerator_dict

# ==============================
# example configs
# ==============================

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
