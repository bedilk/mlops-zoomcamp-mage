from typing import Tuple

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlops.utils.data_preparation.cleaning import clean
from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.data_preparation.feature_selector import select_features
from mlops.utils.data_preparation.splitters import split_on_value

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(df: pd.DataFrame, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_on_feature = kwargs.get('split_on_feature')
    split_on_feature_value = kwargs.get('split_on_feature_value')
    target = kwargs.get('target')

    df = clean(df)
    df = combine_features(df)
    df = select_features(df, features=[split_on_feature, target])

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    train_dicts = df[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    print(f'Mean Squared Error: {mean_squared_error(y_train, y_pred)}')

    df_train, df_val = split_on_value(
        df,
        split_on_feature,
        split_on_feature_value,
    )

    return df, df_train, df_val