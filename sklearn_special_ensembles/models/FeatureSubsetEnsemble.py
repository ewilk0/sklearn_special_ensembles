import copy
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm


class FeatureSubsetEnsemble():
    """
    An ensemble to train estimators on different features of the data.
    This appraoch injects strong diversity into the ensemble and is
    particularly helpful for hedging bets against noisy data.
    """


    def __init__(self, base_estimator, verbose: int = -1):
        """
        Initializes the class.

        :param base_estimator: The base estimator to fit to the different 
            feature subsets of the data.
        :param verbose: -1 to suppress logs. Any other value to allow them.
        """

        self.base_estimator = copy.deepcopy(base_estimator)

        self.columns_to_estimators = {}

        self.verbose = verbose
    
    
    def fit(self, 
            inputs: pd.DataFrame, 
            targets: pd.DataFrame,
            train_col_groups: Iterable) -> None:
        """
        Fits the ensemble to the input and target DataFrames.

        :param inputs: The inputs.
        :param targets: The targets.
        :param train_col_groups: Iterable of the feature groups to train each model on.
        """

        if isinstance(targets, pd.DataFrame):
            if self.verbose != -1:
                print("[FeatureSubsetEnsemble] Converting `targets` to pandas Series")
            targets = pd.Series(targets)

        assert len(inputs) == len(targets), "Inputs and targets must be of the same length"

        for col_group in train_col_groups:
            for col in col_group:
                assert col in inputs.columns, f"Column {col} not in columns of input df"
        
        if self.verbose != -1:
            looper = tqdm(train_col_groups)
            looper.set_description("[FeatureSubsetEnsemble] Fitting models to col groups...")
        else:
            looper = train_col_groups
        for col_group in looper:
            these_inputs = inputs[list(col_group)]
            self.base_estimator.fit(these_inputs, targets)
            self.columns_to_estimators[tuple(col_group)] = copy.deepcopy(self.base_estimator)
        
    
    def predict(self, 
                inputs: pd.DataFrame, 
                estimator_weights: Iterable = None) -> np.array:
        """
        Predicts a given set of inputs.

        :param inputs: The inputs.
        :param estimator_weights: The weights on the estimators in the ensemble. Uniform
            if none is specified.

        :return preds: The predictions.
        """

        if estimator_weights is not None:
            estimator_weights = list(estimator_weights)

        num_estimators = len(self.columns_to_estimators.keys())

        if estimator_weights is not None and len(estimator_weights) != num_estimators:
            warnings.warn(f"Ensemble was fit with {num_estimators} models but \
                          has been given {len(estimator_weights)} weights. Taking first \
                          {num_estimators} weights only.")
            estimator_weights = estimator_weights[:num_estimators]
        
        if estimator_weights is None:
            if self.verbose != -1:
                print("[FeatureSubsetEnsemble] Setting estimator weights to be uniform.")
            estimator_weights = [1/num_estimators for _ in range(num_estimators)]
        else:
            estimator_weights = [weight/sum(estimator_weights) for weight in estimator_weights]

        inputs["preds"] = 0

        for i, col_group in enumerate(self.columns_to_estimators.keys()):
            these_cols = list(col_group)
            this_df = inputs[these_cols]
            this_estimator = self.columns_to_estimators[col_group]
            these_preds = this_estimator.predict(this_df)
            inputs["preds"] += (these_preds * estimator_weights[i])
        
        preds = inputs["preds"].to_numpy()
        inputs = inputs.drop(columns=["preds"])

        return preds