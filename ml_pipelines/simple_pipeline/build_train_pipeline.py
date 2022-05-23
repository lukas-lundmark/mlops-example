#!/usr/bin/env python3
import argparse
from pathlib import Path
from azureml.core.workspace import Workspace
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.core.runconfig import RunConfiguration

from ml_pipelines.utils import (
    EnvironmentVariables,
    get_aml_compute,
    get_environment,
)

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline-id-output", default=None)
arguments = parser.parse_args()
pipeline_id_output = arguments.pipeline_id_output

ws = Workspace.from_config()
env_vars = EnvironmentVariables()

cpu_cluster = get_aml_compute(ws, env_vars)
target_env = get_environment(ws, env_vars)

run_config = RunConfiguration()
run_config.environment = target_env

training_dataset_name = PipelineParameter(
    name="train_ds_name", default_value=env_vars.train_ds
)
test_dataset_name = PipelineParameter(
    name="test_ds_name", default_value=env_vars.train_ds
)
model_name = PipelineParameter(name="model_name", default_value=env_vars.model_name)
train_step = PythonScriptStep(
    name="test_train_model",
    script_name="train_pipeline.py",
    source_directory="src",
    compute_target=cpu_cluster,
    runconfig=run_config,
    allow_reuse=False,
    arguments=[
        "--ds-train",
        training_dataset_name,
        "--ds-test",
        test_dataset_name,
        "--model-name",
        model_name
    ],
)

training_pipeline_name = env_vars.train_pipeline_name
pipeline_endpoint_name = training_pipeline_name

pipeline = Pipeline(
    workspace=ws, steps=[train_step], description="Model Training and Deployment"
)
pipeline.validate()
published_pipeline = pipeline.publish(training_pipeline_name)
pipeline_id = published_pipeline.id

if pipeline_id_output is not None:
    Path(pipeline_id_output).write_text(pipeline_id)
else:
    print(pipeline_id)
