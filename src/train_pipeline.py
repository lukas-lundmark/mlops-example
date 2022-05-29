from azureml.core import Run, Model, Dataset
import argparse
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
from pathlib import Path


def prepare_data(df):
    df = df.copy()
    y = df.pop("price")
    return df, y


def main():
    run = Run.get_context()
    ws = run.experiment.workspace

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds-train", help="Name of training dataset")
    parser.add_argument("--ds-test", help="Name of test dataset")
    parser.add_argument("--model-name", help="Name of model register")
    arguments = parser.parse_args()
    ds_train = arguments.ds_train
    ds_test = arguments.ds_test
    model_name = arguments.model_name

    assert ds_test in ws.datasets and ds_train in ws.datasets
    ds_train = Dataset.get_by_name(ws, name=ds_train)
    ds_test = Dataset.get_by_name(ws, name=ds_test)
    X_train, y_train = prepare_data(ds_train.to_pandas_dataframe())
    X_test, y_test = prepare_data(ds_test.to_pandas_dataframe())

    regressor = LinearRegression()
    ct = make_column_transformer(
        (MaxAbsScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(), make_column_selector(dtype_include=object)),
    )

    model = Pipeline([("ColumnTransformer", ct), ("Regressor", regressor)])
    model.fit(X_train, y_train)
    y_ = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_, squared=False)
    r2 = r2_score(y_test, y_)

    print("rmse", rmse)
    print("r2", r2)

    run.log("rmse", rmse)
    run.log("r2", r2)
    run.parent.log("rmse", rmse)
    run.parent.log("r2", r2)

    path = Path("outputs", "model.pkl")
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, filename=str(path))

    all_models = Model.list(ws, name=model_name)
    if all(rmse < float(model.tags.get("rmse", np.inf)) for model in all_models):
        print("Found a new winner. Registering the model.")
        run.upload_file(str(path.name), path_or_stream=str(path))
        run.register_model(
            model_name=model_name,
            model_path=str(path.name),
            description="Linear Diamond Regression Model",
            model_framework="ScikitLearn",
            datasets=[
                ("training dataset", ds_train),
                ("test dataset", ds_test),
            ],
            tags={"rmse": rmse, "r2": r2},
        )
    else:
        print("Model did not improve result. Cancelling run.")
        run.parent.cancel()


if __name__ == "__main__":
    main()
