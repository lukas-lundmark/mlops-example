#!/usr/bin/env python3
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.linear_model import LinearRegression

from azureml.core import Run

import numpy as np
import joblib
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--model-name", required=True)
arguments = parser.parse_args()
output = arguments.output
model_name = arguments.model_name


def build_model():
    regressor = LinearRegression()
    ct = make_column_transformer(
        (MaxAbsScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(), make_column_selector(dtype_include=object)),
    )
    pipeline = Pipeline([("ColumnTransformer", ct), ("Regressor", regressor)])
    return pipeline


def prepare_dataset(df):
    y = df.pop("price")
    return df, y


def compute_metrics(y, y_, metrics_fn):
    return {name: fn(y, y_) for name, fn in metrics_fn.items()}


run = Run.get_context()
workspace = run.experiment.workspace
train_df = run.input_datasets["train_ds"].to_pandas_dataframe()
print("Train Dataframe", train_df)
print("Train Dataframe", train_df.dtypes)
test_df = run.input_datasets["test_ds"].to_pandas_dataframe()
print("Test Dataframe", test_df)

model = build_model()
model.fit(*prepare_dataset(train_df))
test_df, y = prepare_dataset(test_df)
y_ = model.predict(test_df)
metrics_funcitons = {
    "rmse": lambda y, y_: mean_squared_error(y, y_, squared=False),
    "r2": r2_score,
    "mae": mean_absolute_error,
}

metrics = compute_metrics(y, y_, metrics_fn=metrics_funcitons)
for name, value in metrics.items():
    run.log(name, value)
    run.parent.log(name, value)

# Write to pipeline output
Path(output).mkdir(exist_ok=True)
joblib.dump(value=model, filename=Path(output, model_name))

# Write to step output so it's logged in run history
Path("outputs").mkdir(exist_ok=True)
joblib.dump(value=model, filename=Path("outputs", model_name))
