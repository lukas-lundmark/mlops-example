#!/usr/bin/env python3
from azureml.core import Workspace
from ml_pipelines.utils import EnvironmentVariables
from azureml.core.webservice import AksWebservice

workspace = Workspace.from_config()
env_vars = EnvironmentVariables()
web_services = AksWebservice(workspace, name=env_vars.service_name)
web_services.delete()
