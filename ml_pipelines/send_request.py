#!/usr/bin/env python3
import requests
import argparse
from pathlib import Path
import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", help="example csv to read records from")
    parser.add_argument("--n", type=int, default=10, help="Send this many examples from the csv")
    parser.add_argument(
        "--url",
        help="Remote URL for the service, usually something like http:<something>/score",
    )
    parser.add_argument(
        "--key", help="Set the key if you enabled key-based authentication"
    )
    parser.add_argument(
        "--output", help="Write response to a file"
    )
    arguments = parser.parse_args()

    n = arguments.n
    url = arguments.url
    source_file = Path(arguments.source_file)
    assert source_file.exists()

    df = pd.read_csv(source_file)
    records = df.iloc[:n].to_dict(orient="records")
    print(f"Sending these records: {records} to {url}")

    headers = {"Content-Type": "application/json"}
    if arguments.key is not None:
        print("Using key authentication")
        headers["Authorization"] = f"Bearer {arguments.key}"

    response = requests.post(url=url, data=json.dumps(records), headers=headers)
    if response.status_code == 200:
        print("Received response successfully")
        response_json = response.json()
        if arguments.output is not None:
            Path(arguments.output).write_text(response_json)
        else:
            print(response_json)
    else:
        print(f"Failed with error code: {response.status_code}")


if __name__ == "__main__":
    main()
