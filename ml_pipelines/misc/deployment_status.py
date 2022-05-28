#!/usr/bin/env python3
from azureml.core import Workspace
from ml_pipelines.utils import EnvironmentVariables
from azureml.core.webservice import Webservice

workspace = Workspace.from_config()
env_vars = EnvironmentVariables()

try:
    service = Webservice(workspace, name=env_vars.service_name)
    print(service.state)
except:
    print("Not deployed")
