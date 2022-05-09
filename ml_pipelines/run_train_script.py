#!/usr/bin/env python3
from azureml.core import Environment, Workspace, Experiment, ScriptRunConfig

experiment_name = "diamond-regression-script"
environment_name = "diamond-environment"

ws = Workspace.from_config()
experiment = Experiment(ws, experiment_name)

try:
    environment = Environment.get(ws, name=environment_name)
except Exception as e:
    print("Defining a new environment")
    environment = Environment.from_conda_specification(
        name=environment_name, file_path="environment_setup/ci_dependencies.yml"
    )
    environment.register(ws)

src = ScriptRunConfig('src', script='train.py', environment=environment)
run = experiment.submit(src)
run.wait_for_completion(show_output=True)
