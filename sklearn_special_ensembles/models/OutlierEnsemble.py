import copy
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

from sklearn_special_ensembles.models.SegmentEnsemble import SegmentEnsemble


class OutlierEnsemble():
    """
    An ensemble that fits a separate model to a user-designated group 
    of "outlier" IDs to allow the other model in the ensemble to capture
    more of the signal in the data by ignoring the noisy entries.
    """
    

    def __init__(self, base_estimator, verbose: int = -1):
        """
        Initializes the class.

        :param base_estimator: The base estimator to fit to the non-outlier and 
            outlier groups.
        :param verbose: -1 to suppress logs. Any other value to allow them.
        """

        self.base_estimator = copy.deepcopy(base_estimator)
        self.segment_ensemble = SegmentEnsemble(base_estimator=self.base_estimator,
                                                verbose=verbose)
        
        self.id_col = None
        self.outlier_ids = None

        self.verbose = verbose

    
    def fit(self, 
            inputs: pd.DataFrame, 
            targets: pd.DataFrame,
            id_col: str,
            outlier_ids: Iterable,
            fit_general_model: bool = True) -> None:
        """
        Fits ensemble to non-outlier and outlier groups based on ID column.

        :param inputs: The inputs.
        :param targets: The targets.
        :param id_col: The column holding the IDs in which to find the outlier-designated groups.
        :param outlier_ids: The iterable of outlier-designated IDs to which to fit a separate model.
        :param target_col: The name of the target column.
        :param fit_general_model: Whether or not to fit a model to all the available data.
        """

        if isinstance(targets, pd.Series):
            targets = pd.DataFrame(targets)
        
        assert id_col in inputs.columns, f"`id_col` {id_col} not found in `inputs`"
        assert len(inputs) == len(targets), "Inputs and targets must be of the same length"

        inputs["outlier_flag"] = inputs[id_col].isin(outlier_ids).astype(int)
        
        if self.verbose != -1:
            print("[OutlierEnsemble] Fitting through SegmentEnsemble...")
        self.segment_ensemble.fit(
            inputs, 
            targets, 
            segment_cols=["outlier_flag"], 
            fit_general_model=fit_general_model
        )
        inputs = inputs.drop(columns=["outlier_flag"])

        self.id_col = id_col
        self.outlier_ids = outlier_ids


    def predict(self, inputs: pd.DataFrame, percent_general_model: float = 0.) -> np.array:
        """
        Predict a given set of inputs.

        :param inputs: The inputs.
        :param percent_general_model: The percentage of the general model to use.
        """

        inputs["outlier_flag"] = inputs[self.id_col].isin(self.outlier_ids).astype(int)

        preds = self.segment_ensemble.predict(inputs, percent_general_model=percent_general_model)

        inputs = inputs.drop(columns=["outlier_flag"])

        return preds