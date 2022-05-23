#!/usr/bin/env python3
import argparse
from azureml.pipeline.core import PublishedPipeline, PipelineEndpoint
from azureml.core import Workspace
from ml_pipelines.utils import EnvironmentVariables

ws = Workspace.from_config()
env_vars = EnvironmentVariables()

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline-id", required=True, help="Published Pipeline to invoke")
arguments = parser.parse_args()

published_pipeline = PublishedPipeline.get(ws, id=arguments.pipeline_id)
try:
    pipeline_endpoint = PipelineEndpoint.get(ws, name=env_vars.pipeline_endpoint_name)
    pipeline_endpoint.add_default(published_pipeline)
except Exception:
    pipeline_endpoint = PipelineEndpoint.publish(
        workspace=ws,
        name=env_vars.pipeline_endpoint_name,
        pipeline=published_pipeline,
        description="Pipeline Endpoint for Departure Prediction",
    )
