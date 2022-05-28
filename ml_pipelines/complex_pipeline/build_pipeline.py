#!/usr/bin/env python3
from pathlib import Path
import argparse
from azureml.core.workspace import Workspace

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.data import OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
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

ignore_pipeline_param = PipelineParameter(name="ignore_data_check", default_value=False)
train_ds_pipeline_param = PipelineParameter(
    name="train_dataset", default_value="diamonds-train"
)
test_ds_pipeline_param = PipelineParameter(
    name="test_dataset", default_value="diamonds-test"
)
model_name_pipeline_param = PipelineParameter(
    name="model_name", default_value=env_vars.model_name
)

check_data_step = PythonScriptStep(
    name="Check if data is updated",
    script_name="check_data.py",
    source_directory="src/complex_pipeline",
    compute_target=cpu_cluster,
    arguments=[
        "--ignore",
        ignore_pipeline_param,
        "--ds-name",
        train_ds_pipeline_param,
        "--model-name",
        model_name_pipeline_param,
    ],
    runconfig=run_config,
    allow_reuse=False,
)

output = OutputFileDatasetConfig()

clean_data_step = PythonScriptStep(
    name="Clean and write data",
    script_name="clean_data.py",
    source_directory="src/complex_pipeline",
    compute_target=cpu_cluster,
    arguments=[
        "--ds-name",
        train_ds_pipeline_param,
        "--test-ds-name",
        test_ds_pipeline_param,
        "--output",
        output,
    ],
    runconfig=run_config,
    allow_reuse=False,
)

train_ds = output.read_parquet_files(path_glob="train*")
test_ds = output.read_parquet_files(path_glob="test*")
train_ds_consumption = DatasetConsumptionConfig("train_ds", train_ds)
test_ds_consumption = DatasetConsumptionConfig("test_ds", test_ds)

model_output = PipelineData("model_output")
train_step = PythonScriptStep(
    name="Train model",
    script_name="train.py",
    source_directory="src/complex_pipeline",
    arguments=[
        "--model-name",
        model_name_pipeline_param,
        "--output",
        model_output,
    ],
    inputs=[train_ds_consumption, test_ds_consumption],
    outputs=[model_output],
    compute_target=cpu_cluster,
    runconfig=run_config,
    allow_reuse=False,
)

key_metric_pipeline_param = PipelineParameter(name="key_metric", default_value="rmse")
lower_better_pipeline_param = PipelineParameter(name="lower_better", default_value=True)
allow_eval_cancel_pipeline_param = PipelineParameter(
    name="allow_eval_cancel", default_value=True
)
build_id_param = PipelineParameter(name="build_id", default_value="")

evaluate_step = PythonScriptStep(
    name="Evaulate model performance",
    script_name="evaluate.py",
    source_directory="src/complex_pipeline",
    arguments=[
        "--key-metric",
        key_metric_pipeline_param,
        "--model-name",
        model_name_pipeline_param,
        "--lower-better",
        lower_better_pipeline_param,
        "--allow_run_cancel",
        allow_eval_cancel_pipeline_param,
    ],
    compute_target=cpu_cluster,
    runconfig=run_config,
    allow_reuse=False,
)

register_step = PythonScriptStep(
    name="Register New Model",
    script_name="register.py",
    source_directory="src/complex_pipeline",
    arguments=[
        "--model-name",
        model_name_pipeline_param,
        "--step-input",
        model_output,
        "--build-id",
        build_id_param,
    ],
    compute_target=cpu_cluster,
    runconfig=run_config,
    inputs=[model_output],
    allow_reuse=False,
)

# Set the order of execution
clean_data_step.run_after(check_data_step)
train_step.run_after(clean_data_step)
evaluate_step.run_after(train_step)
register_step.run_after(evaluate_step)

pipeline = Pipeline(
    workspace=ws,
    steps=[check_data_step, clean_data_step, train_step, evaluate_step, register_step],
    description="Model Training and Deployment",
)
pipeline.validate()
published_pipeline = pipeline.publish(env_vars.train_pipeline_name)
pipeline_id = published_pipeline.id

if pipeline_id_output is not None:
    Path(pipeline_id_output).write_text(pipeline_id)
else:
    print(pipeline_id)
