#!/usr/bin/env python3

import os
import joblib
import json
import pandas as pd
import os
from sklearn import *
import logging
from pathlib import Path
import logging

# Local import
from data.cast import to_dataframe, convert_dtypes

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def init():
    global model
    models = list(Path(os.getenv("AZUREML_MODEL_DIR")).glob("*"))
    if len(models) == 0:
        raise ValueError("No model found")

    model_path = str(models[0])
    logger.info("Attempting to load model at %s", model_path)
    model = joblib.load(model_path)
    logger.info("Successfully loaded model at %s", model_path)


def run(raw_data):
    logger.info("Received this json string: %s", raw_data)
    records = json.loads(raw_data)
    logger.info("Successfully loaded data from json")

    # Make sure that inputed data is correct
    df = to_dataframe(records)
    df = convert_dtypes(df)

    logger.info("Successfully converted objects to Pandas dataframe")
    response = model.predict(df)
    df["predicted_price"] = response
    return json.dumps(df.to_dict(orient="records"))
