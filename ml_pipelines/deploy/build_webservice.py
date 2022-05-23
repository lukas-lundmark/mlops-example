#!/usr/bin/env python3
from azureml.core import Model, Workspace
from azureml.core.model import InferenceConfig
from argparse import ArgumentParser
from pathlib import Path

from ml_pipelines.utils import EnvironmentVariables, get_environment

workspace = Workspace.from_config()
env_vars = EnvironmentVariables()

parser = ArgumentParser("Build Scoring Image")
parser.add_argument("--model-version", default=None)
parser.add_argument("--file-output", default="./scoring-image-output.txt")
args = parser.parse_args()

environment = get_environment(workspace, env_vars)
inference_config = InferenceConfig(
    source_directory=env_vars.scoring_dir,
    entry_script=env_vars.scoring_file,
    environment=environment,
)

# Download and Package the model
model = Model(workspace, name=env_vars.model_name, version=args.model_version)
package = Model.package(workspace, models=[model], inference_config=inference_config)
package.wait_for_creation(show_output=True)

output_file = Path(args.file_output)
output_file.parent.mkdir(exist_ok=True)
output_file.write_text(package.location)
