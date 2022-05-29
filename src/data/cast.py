#!/usr/bin/env python3
import pandas as pd


FLOAT_COLS = ("carat", "depth", "table", "price", "x", "y", "z")
CAT_COLS = ("cut", "color", "clarity")


def to_dataframe(records):
    df = pd.DataFrame(records)
    assert all(col in df.columns for col in (list(FLOAT_COLS) + list(CAT_COLS)))
    return df


def convert_dtypes(df):
    """Make sure that the input columns has the correct dtypes"""
    float_dict = {col: float for col in FLOAT_COLS}
    cat_dict = {col: object for col in CAT_COLS}
    convert_dict = {**float_dict, **cat_dict}

    df = df.astype(convert_dict)
    return df[list(FLOAT_COLS) + list(CAT_COLS)]
