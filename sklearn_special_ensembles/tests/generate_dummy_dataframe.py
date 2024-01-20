from typing import Iterable

import numpy as np
import pandas as pd


def generate_dummy_dataframe(
        num_observations: int = 100,
        percent_train: float = 0.8,
        num_numerical_predictors: int = 2,
        num_categorical_predictors: int = 1,
        categories_by_column: Iterable = [[1, 2, 3]]
) -> (pd.DataFrame, pd.DataFrame):
    """
    Generates dummy train and test DataFrames for tests.

    :param num_observations: The number of rows in the df.
    :param percent_train: The percentage of observations for the train split.
    :param num_numerical_predictors: The number of numerical predictors in the df.
    :param num_categorical_predictors: The number of categorical predictors in the df.
    :param categories_by_column: An iterable of the allowed categories for each cat col.

    :return train_df, test_df: The dummy train and test DataFrames.
    """

    assert len(categories_by_column) == num_categorical_predictors

    # generate the values
    if num_numerical_predictors > 0:
        numerical_predictor_vals = np.random.random((num_observations, num_numerical_predictors))
    if num_categorical_predictors > 0:
        categorical_predictor_vals = np.zeros((num_observations, num_categorical_predictors))
        for i in range(num_categorical_predictors):
            these_cats = categories_by_column[i]
            this_cat_col = np.random.choice(these_cats, size=num_observations, replace=True)
            categorical_predictor_vals[:, i] = this_cat_col
    
    target_vals = np.random.random((num_observations, 1))
    if num_numerical_predictors > 0:
        all_vals = numerical_predictor_vals
    if num_categorical_predictors > 0:
        all_vals = np.hstack([all_vals, categorical_predictor_vals])
    all_vals = np.hstack([all_vals, target_vals])

    numerical_cols = [f"numerical_{i}" for i in range(num_numerical_predictors)]
    cat_cols = [f"categorical_{i}" for i in range(num_categorical_predictors)]

    df = pd.DataFrame(all_vals, columns=numerical_cols+cat_cols+["target"])
    df[cat_cols] = df[cat_cols].astype("category")
    train_df = df[:int(num_observations*percent_train)]
    test_df = df[int(num_observations*percent_train):]

    return train_df, test_df