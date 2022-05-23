#!/usr/bin/env python3
import argparse
from azureml.pipeline.core import PublishedPipeline
from azureml.core import Experiment, Workspace
from ml_pipelines.utils import EnvironmentVariables

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pipeline-id",
    help="Specific Pipeline ID to run. If not set the latest version of the training pipeline will run.",
)
arguments = parser.parse_args()

ws = Workspace.from_config()
env_vars = EnvironmentVariables()

if arguments.pipeline_id is None:
    # Get the pipelines in the workspace and find the latest version of our pipeline
    pipelines = PublishedPipeline.list(workspace=ws)
    pipelines = [p for p in pipelines if p.name == env_vars.train_pipeline_name]
    # The pipelines are generally in the reverse order they were created
    published_pipeline = pipelines[0]
else:
    published_pipeline = PublishedPipeline.get(ws, id=arguments.pipeline_id)

print("Started run using pipeline: ", published_pipeline.id)
experiment = Experiment(ws, env_vars.experiment_name)
run = experiment.submit(published_pipeline)
run.wait_for_completion(show_output=True)
