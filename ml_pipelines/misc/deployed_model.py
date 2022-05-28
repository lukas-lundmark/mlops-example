#!/usr/bin/env python3
"""Prints the model that is currently deployed to the webservice
"""

from azureml.core import Workspace
from ml_pipelines.utils import EnvironmentVariables
from azureml.core.webservice import Webservice

workspace = Workspace.from_config()
env_vars = EnvironmentVariables()

service = Webservice(workspace, name=env_vars.service_name)

# Get the models registered with the service
models = service.models
models = sorted(
    (model for model in models if model.name == env_vars.model_name),
    key=lambda x: x.version,
    reverse=True,
)

if len(models) == 0:
    raise ValueError("No model found")

best_model = models[0]
print(best_model.id)
