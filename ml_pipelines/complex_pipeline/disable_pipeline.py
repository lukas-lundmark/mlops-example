#!/usr/bin/env python3
import argparse
from azureml.pipeline.core import PublishedPipeline
from azureml.core import Workspace
from ml_pipelines.utils import EnvironmentVariables

ws = Workspace.from_config()
env_vars = EnvironmentVariables()

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline-id", required=True, help="Published Pipeline to invoke")
arguments = parser.parse_args()

published_pipeline = PublishedPipeline.get(ws, id=arguments.pipeline_id)
published_pipeline.disable()
