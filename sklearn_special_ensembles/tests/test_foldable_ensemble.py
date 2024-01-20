import copy
import sys, os
sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath("../models/FoldableEnsemble"))))

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from generate_dummy_dataframe import generate_dummy_dataframe
from models.FoldableEnsemble import FoldableEnsemble


n_splits = 4

# create a dummy dataframe
train_df, test_df = generate_dummy_dataframe(
    num_categorical_predictors=1, 
    categories_by_column=[[1, 2, 3, 4]]
)

# create the model
base_model = LGBMRegressor(verbose=-1)
foldable_ensemble = FoldableEnsemble(
    estimators=[copy.deepcopy(base_model) for _ in range(n_splits)], 
    verbose=0
)

foldable_ensemble.fit(
    train_df.drop(columns=["target"]),
    train_df["target"],
    split_by_folds=True
)
preds = foldable_ensemble.predict(test_df.drop(columns=["target"]))

assert preds.shape == (len(test_df),)