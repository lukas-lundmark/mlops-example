#!/usr/bin/env python3

from azureml.core import Workspace
from azureml.pipeline.core import PipelineEndpoint
from ml_pipelines.utils import EnvironmentVariables, get_experiment

ws = Workspace.from_config()
env_vars = EnvironmentVariables()

pipeline_endpoint = PipelineEndpoint.get(ws, name=env_vars.train_pipeline_name)
experiment = get_experiment(ws, env_vars)
run = experiment.submit(
    pipeline_endpoint, tags={"endpoint_version": pipeline_endpoint.default_version}
)
run.wait_for_completion(show_output=True)
