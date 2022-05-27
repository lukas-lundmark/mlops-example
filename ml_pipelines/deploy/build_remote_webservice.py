#!/usr/bin/env python3
from azureml.core import Model, Workspace
from azureml.core.model import InferenceConfig
from argparse import ArgumentParser
from azureml.core.webservice import AksWebservice, LocalWebservice
from ml_pipelines.utils import EnvironmentVariables, get_environment, get_aks_cluster
from pathlib import Path

workspace = Workspace.from_config()
env_vars = EnvironmentVariables()

parser = ArgumentParser("Build Scoring Image")
parser.add_argument("--model-version", default=None)
parser.add_argument("--local", action='store_true')
parser.add_argument("--url-output", default=None)
args = parser.parse_args()

environment = get_environment(workspace, env_vars)
inference_config = InferenceConfig(
    entry_script=env_vars.scoring_file,
    source_directory=env_vars.scoring_dir,
    environment=environment,
)

model = Model(workspace, name=env_vars.model_name, version=args.model_version)

if not args.local:
    deployment_config = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
    aks_target = get_aks_cluster(workspace, env_vars)
    print(aks_target)
    kwargs = {'deployment_target': aks_target}
else:
    deployment_config = LocalWebservice.deploy_configuration(port=6789)
    kwargs = {}

service = Model.deploy(
    workspace=workspace,
    name=env_vars.service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True,
    **kwargs
)
service.wait_for_deployment(show_output=True)
if args.url_output is not None:
    Path(args.url_output).write_text(service.scoring_uri)
    f"SCORING_URL={service.scoring_uri}\n"
