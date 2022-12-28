
"""Vertex pipeline configurations."""

import os


PROJECT_ID = os.getenv("PROJECT_ID", "")
LOCATION = os.getenv("LOCATION", "us-central1")
BUCKET = os.getenv("BUCKET", "")

INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "n1-highmem-64")
CPU_LIMIT = os.getenv("CPU_LIMIT", "64")
MEMORY_LIMIT = os.getenv("MEMORY_LIMIT", "416")
GPU_LIMIT = os.getenv("GPU_LIMIT", "4")
GPU_TYPE = os.getenv("GPU_TYPE", "NVIDIA_TESLA_T4")

MACHINE_TYPE = os.getenv("MACHINE_TYPE", "a2-highgpu-4g")
REPLICA_COUNT = os.getenv("REPLICA_COUNT", "1")
ACCELERATOR_TYPE = os.getenv("ACCELERATOR_TYPE", "NVIDIA_TESLA_A100")
ACCELERATOR_NUM = os.getenv("ACCELERATOR_NUM", "4")
NUM_WORKERS = os.getenv("NUM_WORKERS", "4")
