#!/usr/bin/env python3
import argparse
from azureml.pipeline.core import PublishedPipeline, PipelineEndpoint
from azureml.core import Experiment, Workspace
from ml_pipelines.utils import EnvironmentVariables

ws = Workspace.from_config()
env_vars = EnvironmentVariables()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pipeline-id",
    default=None,
    help="Id of published pipeline to invoke. Ignore to invoke the default pipeline endpoint instead",
)
parser.add_argument(
    "--ignore-data-check",
    action="store_true",
    help="Run the given pipeline regardless if new data is available or not",
)

arguments = parser.parse_args()

if arguments.pipeline_id is None:
    published_pipeline = PipelineEndpoint.get(ws, name=env_vars.pipeline_endpoint_name)
else:
    published_pipeline = PublishedPipeline.get(ws, id=arguments.pipeline_id)

experiment = Experiment(ws, env_vars.experiment_name)

pipeline_parameters = {}
if arguments.ignore_data_check:
    pipeline_parameters.update({"ignore_data_check": "True"})

run = experiment.submit(published_pipeline, pipeline_parameters=pipeline_parameters)
run.wait_for_completion(show_output=True)
