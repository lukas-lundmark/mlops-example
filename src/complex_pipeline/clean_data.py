#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from azureml.core import Dataset, Run


parser = argparse.ArgumentParser()
parser.add_argument("--ds-name", required=True, help="Name of the training dataset")
parser.add_argument("--test-ds-name", required=True, help="Name of the test dataset")
parser.add_argument("--output", required=True, help="The output file to save the ")

arguments = parser.parse_args()

run = Run.get_context()
workspace = run.experiment.workspace

assert arguments.ds_name in workspace.datasets
assert arguments.test_ds_name in workspace.datasets

train_ds = Dataset.get_by_name(workspace, arguments.ds_name)
test_ds = Dataset.get_by_name(workspace, arguments.test_ds_name)

# Add the datasets to the run
tags = {"train_ds": train_ds.id, "test_ds": test_ds.id}
run.set_tags(tags)
run.parent.set_tags(tags)

def clean(df):
    df_ = df.copy()
    # Filter out the zero values we observed before
    df_ = df_[~((df_['x'] == 0) | (df_['y'] == 0) | (df_['z'] == 0))]
    df_ = df_.dropna()
    return df

train_df = clean(train_ds.to_pandas_dataframe())
test_df = clean(test_ds.to_pandas_dataframe())

output_path = arguments.output
Path(output_path).mkdir(parents=True, exist_ok=True)

with open(Path(output_path, 'train.pt'), 'wb') as fh:
    train_df.to_parquet(fh, index=False)

with open(Path(output_path, 'test.pt'), 'wb') as fh:
    test_df.to_parquet(fh, index=False)
