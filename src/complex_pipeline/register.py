#!/usr/bin/env python3

from pathlib import Path
from azureml.core import Dataset, Run
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name",
    type=str,
    help="Name of the Model",
)

parser.add_argument("--step-input", type=str, help="input from previous steps")
arguments = parser.parse_args()
model_path = arguments.step_input
model_name = arguments.model_name


def get_information(run):
    parent_tags = run.parent.tags
    workspace = run.experiment.workspace

    train_dataset_id = parent_tags["train_ds"]
    test_dataset_id = parent_tags["test_ds"]

    metrics = run.parent.get_metrics()
    datasets = [
        ("train_ds", Dataset.get_by_id(workspace, train_dataset_id)),
        ("test_ds", Dataset.get_by_id(workspace, test_dataset_id)),
    ]
    return metrics, datasets


run = Run.get_context()
metrics, datasets = get_information(run)
run.parent.upload_file(model_name, str(Path(model_path, model_name)))
run.parent.register_model(
    model_name=model_name,
    tags=metrics,
    datasets=datasets,
)
