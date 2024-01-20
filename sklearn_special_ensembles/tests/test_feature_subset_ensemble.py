import sys, os
sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath("../models/FeatureSubsetEnsemble"))))

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from generate_dummy_dataframe import generate_dummy_dataframe
from models.FeatureSubsetEnsemble import FeatureSubsetEnsemble


# create a dummy dataframe
train_df, test_df = generate_dummy_dataframe(
    num_categorical_predictors=2, 
    categories_by_column=[[1, 2, 3, 4], [5, 6]]
)

# create the model
base_model = LGBMRegressor(verbose=-1)
feature_ensemble = FeatureSubsetEnsemble(base_estimator=base_model, verbose=1)

feature_ensemble.fit(
    train_df.drop(columns=["target"]),
    train_df["target"],
    train_col_groups=[["numerical_0", "numerical_1"],
                      ["categorical_0", "categorical_1"]]
)
preds = feature_ensemble.predict(test_df.drop(columns=["target"]))


assert preds.shape == (len(test_df),)