from azureml.core import Model, Workspace
from pathlib import Path
from ml_pipelines.utils import EnvironmentVariables
import json
import argparse

workspace = Workspace.from_config()
parser = argparse.ArgumentParser()
parser.add_argument("--run-id", help="The run id that created the model")
parser.add_argument("--output", required=True, help="Write model information to this file")
arguments = parser.parse_args()

env_vars = EnvironmentVariables()
run_id = env_vars.run_id

if arguments.run_id is not None:
    run_id = arguments.run_id
if run_id is None:
    raise ValueError("No value set for Run ID")

models = Model.list(workspace, name=env_vars.model_name, tags=['buildId', run_id])
n_models = len(models)

if n_models == 0:
    raise ValueError("No model failed for this run")
elif n_models > 1:
    raise ValueError(f"Too many models with this run {n_models}")

Path(arguments.output).write_text(json.dumps(models[0]))
