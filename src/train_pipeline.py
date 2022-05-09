from azureml.core import Run, Model
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
    y = df.pop('price')
    return df, y

def main():
    run = Run.get_context()
    ws = run.experiment.workspace
    datasets = run.input_datasets

    train_dataset = datasets["train_ds"]
    test_dataset = datasets["test_ds"]

    model_name="diamond-linear-regressor-new-debug"

    X_train, y_train = prepare_data(train_dataset.to_pandas_dataframe())
    X_test, y_test  = prepare_data(test_dataset.to_pandas_dataframe())

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

    run.parent.log("rmse", rmse)
    run.parent.log("r2", r2)

    path = Path("output", "model.pkl")
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, filename=str(path))

    run.upload_file(str(path.name), path_or_stream=str(path))

    all_models = Model.list(ws, name=model_name)
    if all(r2 >  float(model.tags.get("r2", -np.inf)) for model in all_models):
        print("Found a new winner. Registering the model.")
        run.register_model(
            model_name=model_name,
            model_path=str(path.name),
            description="Linear Diamond Regression Model",
            model_framework="ScikitLearn",
            datasets=[("training dataset", train_dataset), ("test dataset", test_dataset)],
            tags={"rmse": rmse, "r2": r2}
        )


if __name__ == '__main__':
    main()
