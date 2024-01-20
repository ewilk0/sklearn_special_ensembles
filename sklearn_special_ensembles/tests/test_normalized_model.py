import sys, os
sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath("../models/NormalizedModel"))))

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from generate_dummy_dataframe import generate_dummy_dataframe
from models.NormalizedModel import NormalizedModel


# create a dummy dataframe
train_df, test_df = generate_dummy_dataframe()

# create the model
base_model = LGBMRegressor(verbose=-1)
normalized_model = NormalizedModel(base_estimator=base_model, verbose=-1)

normalized_model.fit(
    train_df.drop(columns=["target"]), 
    train_df["target"], 
    normalizing_col="numerical_0", 
    how="divide"
)
preds = normalized_model.predict(test_df.drop(columns=["target"]))


assert preds.shape == (len(test_df),)