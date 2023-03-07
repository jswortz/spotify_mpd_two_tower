def prepare_worker_pool_specs(
    image_uri,
    args,
    cmd,
    replica_count=1,
    machine_type="n1-standard-16",
    accelerator_count=1,
    accelerator_type="ACCELERATOR_TYPE_UNSPECIFIED",
    reduction_server_count=0,
    reduction_server_machine_type="n1-highcpu-16",
    reduction_server_image_uri="us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver:latest",
):

    if accelerator_count > 0:
        machine_spec = {
            "machine_type": machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
        }
    else:
        machine_spec = {"machine_type": machine_type}

    container_spec = {
        "image_uri": image_uri,
        "args": args,
        "command": cmd,
    }

    chief_spec = {
        "replica_count": 1,
        "machine_spec": machine_spec,
        "container_spec": container_spec,
    }

    worker_pool_specs = [chief_spec]
    if replica_count > 1:
        workers_spec = {
            "replica_count": replica_count - 1,
            "machine_spec": machine_spec,
            "container_spec": container_spec,
        }
        worker_pool_specs.append(workers_spec)
    if reduction_server_count > 1:
        workers_spec = {
            "replica_count": reduction_server_count,
            "machine_spec": {
                "machine_type": reduction_server_machine_type,
            },
            "container_spec": {"image_uri": reduction_server_image_uri},
        }
        worker_pool_specs.append(workers_spec)

    return worker_pool_specs