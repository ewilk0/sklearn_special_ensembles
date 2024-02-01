import sys, os
sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath("../models/SegmentEnsemble"))))

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from generate_dummy_dataframe import generate_dummy_dataframe
from models.SegmentEnsemble import SegmentEnsemble


# create a dummy dataframe
train_df, test_df = generate_dummy_dataframe(
    num_categorical_predictors=2, 
    categories_by_column=[[1, 2], [3, 4]]
)

# create the model
base_model = LGBMRegressor(verbose=-1)
segment_ensemble = SegmentEnsemble(base_estimator=base_model, verbose=1)

segment_ensemble.fit(
    train_df.drop(columns=["target"]), 
    train_df["target"],
    segment_cols=["categorical_0", "categorical_1"]
)
preds = segment_ensemble.predict(test_df.drop(columns=["target"]))


assert preds.shape == (len(test_df),)