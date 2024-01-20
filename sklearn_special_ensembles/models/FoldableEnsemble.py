import copy
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm


class FoldableEnsemble():
    """
    An ensemble that can easily train a set of estimators each on different folds of the
    data and combine their predictions during inference.

    As a rule of thumb, if using the default folds, one should only use this ensemble with 
    >= 3 estimators.
    """


    def __init__(self, estimators, verbose: int = -1):
        """
        Initializes the class.

        :param estimators: The estimators to be fit to the different folds.
        """

        self.estimators = copy.deepcopy(estimators)

        self.verbose = verbose

    
    def fit(self, 
            inputs: pd.DataFrame, 
            targets: pd.Series, 
            split_by_folds: bool = True,
            fold_indices: Iterable = None) -> None:
        """
        Fits the ensemble to input and target DataFrames.

        :param inputs: The inputs.
        :param targets: The targets.
        :param split_by_folds: Whether or not to train models on separate folds.
        :param fold_indices: If `split_by_folds` is true, these will be used as the indices
            for the folds of the inputs on which to train each estimator.
        """

        if isinstance(targets, pd.DataFrame):
            targets = pd.Series(targets)
        
        target_col = targets.name

        assert (inputs.index == targets.index).all(), "Input df and target series must have the same index"

        df = pd.concat([inputs, targets], axis=1)

        if split_by_folds:
            num_splits = len(self.estimators)
            i = 0

            if fold_indices is None:
                if self.verbose != -1:
                    print("[FoldableEnsemble] Setting fold indices to defaults...")
                fold_indices = [tup[0] for tup in KFold(n_splits=num_splits, shuffle=False).split(df)]
            
            if self.verbose != -1:
                looper = tqdm(fold_indices)
                looper.set_description("[FoldableEnsemble] Fitting models...")
            else:
                looper = fold_indices
            
            for train_idx in looper:
                these_inputs = df.loc[train_idx, :].drop(columns=[target_col])
                these_targets = df.loc[train_idx, target_col]
                self.estimators[i].fit(these_inputs, these_targets)
                i += 1

        else:
            if fold_indices is not None:
                warnings.warn(f"`split_by_folds` is False, but fold indices were specified. \
                              Ignoring fold_indices...")

            for i in range(len(self.estimators)):
                self.estimators[i].fit(inputs, targets)
    

    def predict(self, 
                inputs: pd.DataFrame, 
                estimator_weights: Iterable = None) -> np.array:
        """
        Creates predictions for a given set of inputs.

        :param inputs: The inputs, which must have the same columns as those given to `fit`.
        :param estimator_weights: The weights for each estimator in the ensemble, if any.
        """

        inputs["preds"] = 0

        if estimator_weights is None:
            if self.verbose != -1:
                print("[FoldableEnsemble] Setting estimator weights to be uniform.")
            estimator_weights = [1/len(self.estimators) for _ in range(len(self.estimators))]
        else:
            estimator_weights = [weight/sum(estimator_weights) for weight in estimator_weights]

        for i, estimator in enumerate(self.estimators):
            these_preds = estimator.predict(inputs.drop(columns=["preds"]))
            inputs["preds"] += (these_preds * estimator_weights[i])
        
        preds = inputs["preds"].to_numpy()
        inputs = inputs.drop(columns=["preds"])

        return preds