import sys, os
sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath("../models/OutlierEnsemble"))))

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from generate_dummy_dataframe import generate_dummy_dataframe
from models.OutlierEnsemble import OutlierEnsemble


# create a dummy dataframe
train_df, test_df = generate_dummy_dataframe(
    num_categorical_predictors=1, 
    categories_by_column=[[1, 2, 3, 4]]
)

# create the model
base_model = LGBMRegressor(verbose=-1)
outlier_ensemble = OutlierEnsemble(base_estimator=base_model, verbose=0)

outlier_ensemble.fit(
    train_df.drop(columns=["target"]), 
    train_df["target"],
    id_col="categorical_0",
    outlier_ids=[1, 3]
)
preds = outlier_ensemble.predict(test_df.drop(columns=["target"]))


assert preds.shape == (len(test_df),)