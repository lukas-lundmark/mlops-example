#!/usr/bin/env python3

# Simple scripts that cancels the run if no new dataset has been created
import argparse
from azureml.core import Run, Model, Dataset
from datetime import datetime
import pytz
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def true_or_false(arg):
    upper = str(arg).upper()
    if "TRUE".startswith(upper):
        return True
    elif "FALSE".startswith(upper):
        return False
    else:
        raise ValueError  # error condition maybe?


parser = argparse.ArgumentParser()
parser.add_argument("--ds-name", help="Name of training dataset", required=True)
parser.add_argument("--model-name", help="Name of model", required=True)
parser.add_argument("--ignore", type=str, default="False", help="")

arguments = parser.parse_args()
ignore = true_or_false(arguments.ignore)
logger.info(arguments)

run = Run.get_context()
workspace = run.experiment.workspace

try:
    model = Model(workspace, name=arguments.model_name)
    last_train_time = model.created_time
except Exception:
    last_train_time = datetime.min.replace(tzinfo=pytz.UTC)

train_ds = Dataset.get_by_name(workspace, arguments.ds_name)
assert train_ds is not None
dataset_changed_time = train_ds.data_changed_time

if dataset_changed_time < last_train_time and (not ignore):
    logger.info("Dataset has not changed since model was trained last time")
    run.parent.cancel()
else:
    if ignore:
        logger.info("Ignoring check")
    else:
        logger.info(f"Dataset was changed at {dataset_changed_time}, continuing...")
