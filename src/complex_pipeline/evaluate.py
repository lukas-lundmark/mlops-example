#!/usr/bin/env python3
from azureml.core import Model, Run
import argparse
import logging

from azureml.exceptions import WebserviceException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, help="Name of the Model", required=True)

parser.add_argument(
    "--allow_run_cancel",
    type=str,
    help="Set this to false to avoid evaluation step from cancelling run after an unsuccessful evaluation",  # NOQA: E501
    default="true",
)


parser.add_argument(
    "--key-metric",
    type=str,
    help="Key metric to compare",
    default="rmse",
)

parser.add_argument(
    "--lower-better",
    type=bool,
    help="Is the metric better if we move lower",
    default=True,
)


run = Run.get_context()
workspace = run.experiment.workspace

arguments = parser.parse_args()
model_name = arguments.model_name
allow_run_cancel = arguments.allow_run_cancel
key_metric = arguments.key_metric
lower_better = arguments.lower_better

try:
    model = Model(workspace, name=model_name, version=None)
except WebserviceException as e:
    logger.info("No model registered with this name")
    exit(0)


# Get metric from the current production model
production_value = model.tags.get(key_metric, None)

# Get the metrics from the parent step
new_value = run.parent.get_metrics().get(key_metric)

if production_value is None or new_value is None:
    logger.info("Could not find the key metric %s", key_metric)
    if allow_run_cancel:
        run.parent.cancel()

if lower_better and float(production_value) > new_value:
    logger.info(
        "New model has lower %s and therefore better %d < %s",
        key_metric,
        new_value,
        production_value,
    )
elif not lower_better and float(production_value) < new_value:
    logger.info(
        "New model has higher %s and therefore better %d > %s",
        key_metric,
        new_value,
        production_value,
    )
else:
    logger.info(
        "New model did not improve result for metric $s: %d (new) vs. %s (old)",
        key_metric,
        new_value,
        production_value,
    )
    if allow_run_cancel:
        run.parent.cancel()
