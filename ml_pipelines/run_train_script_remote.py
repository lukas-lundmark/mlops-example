#!/usr/bin/env python3
from azureml.core import (
    Environment,
    Workspace,
    Experiment,
    ScriptRunConfig,
)

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

experiment_name = "diamond-regression-script-remote"
environment_name = "diamond-environment"

ws = Workspace.from_config()
experiment = Experiment(ws, experiment_name)

try:
    environment = Environment.get(ws, name=environment_name)
except Exception as e:
    print("Defining a new environment")
    environment = Environment.from_conda_specification(
        name=environment_name, file_path="environment_setup/ci_dependencies.yml"
    )
    environment.register(ws)

cpu_cluster_name = "cpucluster"

try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D2_V2",
        max_nodes=1,
        idle_seconds_before_scaledown=300,  # Scale down after 5 minutes
    )
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)


src = ScriptRunConfig(
    "src", script="train.py", environment=environment, compute_target=cpu_cluster
)
run = experiment.submit(src)
run.wait_for_completion(show_output=True)
