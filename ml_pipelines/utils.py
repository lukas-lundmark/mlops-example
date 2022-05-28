#!/usr/bin/env python3

import os
from dataclasses import dataclass

from azureml.core.compute import ComputeTarget, AmlCompute, AksCompute
from dotenv import load_dotenv
from typing import Optional

from azureml.core import Environment, Workspace, Experiment
from azureml.core.compute_target import ComputeTargetException


@dataclass(frozen=True)
class EnvironmentVariables:
    """Utility class that loads environment variables"""

    # Load the system environment variables
    load_dotenv()

    cpucluster: Optional[str] = os.environ.get("CPUCLUSTER", "cpucluster")
    n_nodes: int = int(os.environ.get("N_NODES", 1))
    vm_size: Optional[str] = os.environ.get("VM_SIZE", "STANDARD_D2_V2")
    idle_limit: int = int(os.environ.get("IDLE_LIMIT", 300))

    environment_name: Optional[str] = os.environ.get(
        "ENVIRONMENT_NAME", "luklun-conda-environment"
    )

    environment_file: Optional[str] = os.environ.get(
        "ENVIRONMENT_FILE", "environment_setup/ci_dependencies.yml"
    )

    scoring_dir: Optional[str] = os.environ.get(
        "SCORING_DIR", "src"
    )

    scoring_file: Optional[str] = os.environ.get(
        "SCORING_FILE", "deployment/score.py"
    )

    service_name: Optional[str] = os.environ.get(
        "SERVICE_NAME", "test-regressor"
    )

    experiment_name: Optional[str] = os.environ.get(
        "EXPERIMENT_NAME", "train-diamond-experiment"
    )

    train_pipeline_name: Optional[str] = os.environ.get(
        "TRAIN_PIPELINE_NAME", "train-pipeline"
    )

    pipeline_endpoint_name: Optional[str] = os.environ.get(
        "PIPELINE_ENDPOINT_NAME", "train-pipeline-endpoint"
    )

    train_ds: Optional[str] = os.environ.get(
        "TRAIN_DS", "diamonds-train"
    )
    test_ds: Optional[str] = os.environ.get(
        "TEST_DS", "diamonds-test"
    )

    model_name: Optional[str] = os.environ.get("MODEL_NAME", "diamond-linear-regressor")
    inference_cluster_name: Optional[str] = os.environ.get("INFERENCE_CLUSTER_NAME", "aks-cluster")

    run_id: Optional[str] = os.environ.get("GITHUB_RUN_ID", None)

def get_aml_compute(ws: Workspace, env_vars: EnvironmentVariables) -> ComputeTarget:
    try:
        cpu_cluster = ComputeTarget(workspace=ws, name=env_vars.cpucluster)
        print("Found existing cluster, use it.")
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=env_vars.vm_size,
            max_nodes=env_vars.n_nodes,
            idle_seconds_before_scaledown=env_vars.idle_limit,
        )
        cpu_cluster = ComputeTarget.create(ws, env_vars.cpucluster, compute_config)

    cpu_cluster.wait_for_completion(show_output=True)
    return cpu_cluster

def get_environment(ws: Workspace, env_vars: EnvironmentVariables):
    environment_name = env_vars.environment_name
    assert environment_name is not None
    try:
        env = Environment.get(ws, name=environment_name)
    except Exception:
        assert env_vars.environment_file is not None
        env = Environment.from_conda_specification(
            name=environment_name, file_path=env_vars.environment_file
        )
    return env

def get_aks_cluster(ws: Workspace, env_vars: EnvironmentVariables) -> AksCompute:
    try:
        aks_target = AksCompute(ws, name=env_vars.inference_cluster_name)
        print("Found AKS Target")
    except ComputeTargetException:
        print("Creating AKS Target")
        provisioning_config = AksCompute.provisioning_configuration(
            vm_size='Standard_D2as_v4',
            agent_count = 1,
            cluster_purpose = AksCompute.ClusterPurpose.DEV_TEST
        )
        aks_target = ComputeTarget.create(
            workspace = ws,
            name = env_vars.inference_cluster_name,
            provisioning_configuration = provisioning_config
        )
        aks_target.wait_for_completion(show_output = True)
    return aks_target

def get_experiment(ws: Workspace, env_vars: EnvironmentVariables) -> Experiment:
    return Experiment(ws, name=env_vars.experiment_name)
