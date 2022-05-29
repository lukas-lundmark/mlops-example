from azureml.core import Model, Workspace
from pathlib import Path
from ml_pipelines.utils import EnvironmentVariables
import argparse

workspace = Workspace.from_config()
parser = argparse.ArgumentParser()
parser.add_argument("--run-id", help="The run id that created the model")
parser.add_argument(
    "--output", required=True, help="Write model information to this file"
)
arguments = parser.parse_args()

env_vars = EnvironmentVariables()
run_id = env_vars.run_id

if arguments.run_id is not None:
    run_id = arguments.run_id

tags = None
if run_id is not None:
    tags = tags = [["buildId", run_id]]

models = Model.list(workspace, name=env_vars.model_name, tags=tags)
n_models = len(models)

if n_models == 0:
    raise ValueError("No matching model found")
elif n_models > 1:
    raise ValueError(f"Too many models: {n_models}, expected 1")

if arguments.output is not None:
    Path(arguments.output).write_text(models[0].id)
else:
    print(models[0].id)
