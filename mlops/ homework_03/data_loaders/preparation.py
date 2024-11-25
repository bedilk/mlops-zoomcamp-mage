from typing import Tuple

import pandas as pd

from mlops.utils.data_preparation.cleaning import clean
from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.data_preparation.feature_selector import select_features
from mlops.utils.data_preparation.splitters import split_on_value


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
import pandas as pd

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def clean(df):
    return read_dataframe(df)

@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_on_feature = kwargs.get('split_on_feature')
    split_on_feature_value = kwargs.get('split_on_feature_value')
    target = kwargs.get('target')

    df = clean(df)
    df = combine_features(df)
    df = select_features(df, features=[split_on_feature, target])

    df_train, df_val = split_on_value(
        df,
        split_on_feature,
        split_on_feature_value,
    )

    return df, df_train, df_val