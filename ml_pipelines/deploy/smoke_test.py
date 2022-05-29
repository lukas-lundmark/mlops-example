import json
from azureml.pipeline import steps
import requests
from azureml.core.webservice import Webservice
from ml_pipelines.utils import EnvironmentVariables
from azureml.core import Workspace, Dataset
from sklearn.metrics import r2_score
import pandas as pd


def send_request(records, uri, key):
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"

    response = requests.post(url=uri, data=json.dumps(records), headers=headers)
    assert response.status_code == 200
    return response.json()


# Connect to the workspace
workspace = Workspace.from_config()
env_vars = EnvironmentVariables()
dataset = Dataset.get_by_name(workspace, name=env_vars.test_ds).to_pandas_dataframe()

# Get the current Webservice
webservice = Webservice(workspace, name=env_vars.service_name)
key, _ = webservice.get_keys()
uri = webservice.scoring_uri

# Send some data from our test dataset to the deployment

total_size = 500
step_size = 10
return_records = []
for i in range(0, total_size, step_size):
    records = dataset.iloc[i : i + step_size].to_dict(orient="records")
    response = send_request(records, uri, key=key)
    return_records.extend(json.loads(response))

# Make sure that the service performs well enough
return_df = pd.DataFrame(return_records)
r2 = r2_score(return_df["predicted_price"], dataset["price"].iloc[:total_size])
print("R2 score", r2)
assert r2 > 0.9
