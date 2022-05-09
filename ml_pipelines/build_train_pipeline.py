#!/usr/bin/env python3
import os
from azureml.core.workspace import Workspace
from dotenv import load_dotenv

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineEndpoint
from azureml.core.runconfig import RunConfiguration
from azureml.core import Dataset
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

from ml_pipelines.utils import (
    EnvironmentVariables,
    get_aml_compute,
    get_environment,
)

dataset_name = "diamonds"

ws = Workspace.from_config()
env_vars = EnvironmentVariables()

train_dataset = Dataset.get_by_name(ws, f"{dataset_name}-train")
test_dataset = Dataset.get_by_name(ws, f"{dataset_name}-test")
train_ds_consumption = DatasetConsumptionConfig("train_ds", train_dataset)
test_ds_consumption = DatasetConsumptionConfig("test_ds", test_dataset)
inputs = [train_ds_consumption, test_ds_consumption]

cpu_cluster = get_aml_compute(ws, env_vars)
target_env = get_environment(ws, env_vars)


run_config = RunConfiguration()
run_config.environment = target_env

train_step = PythonScriptStep(
    name="test_train_model",
    script_name="train_pipeline.py",
    source_directory="src",
    compute_target=cpu_cluster,
    runconfig=run_config,
    inputs=inputs,
    allow_reuse=False,
)

training_pipeline_name = env_vars.train_pipeline_name
pipeline_endpoint_name = training_pipeline_name

pipeline = Pipeline(
    workspace=ws, steps=[train_step], description="Model Training and Deployment"
)
pipeline.validate()

published_pipeline = pipeline.publish(training_pipeline_name)

try:
    pipeline_endpoint = PipelineEndpoint.get(ws, name=pipeline_endpoint_name)
    pipeline_endpoint.add_default(published_pipeline)
except Exception:
    pipeline_endpoint = PipelineEndpoint.publish(
        workspace=ws,
        name=pipeline_endpoint_name,
        pipeline=published_pipeline,
        description="Pipeline Endpoint for Departure Prediction",
    )
