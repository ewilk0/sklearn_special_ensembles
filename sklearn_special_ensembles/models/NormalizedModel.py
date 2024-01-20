import copy
import warnings

import numpy as np
import pandas as pd


class NormalizedModel():
    """
    A model to predict a target normalized by some other column in the data, e.g.,
    dividing solar cell power output by installed capacity. Handles missing values
    by fitting a separate model to non-normalizable rows.
    """


    def __init__(self, base_estimator, verbose: int = -1):
        """
        Initializes the class.

        :param base_estimator: An sklearn model with fit/predict capabilities.
        :param verbose: -1 to suppress logs. Any other value to allow them.
        """

        self.base_estimator = base_estimator
        self.normalizing_estimator = copy.deepcopy(base_estimator)

        self.verbose = verbose

        self.normalizing_col = None
        self.normalize_how = None
    

    def fit(self, 
            inputs: pd.DataFrame, 
            targets: pd.DataFrame, 
            normalizing_col: str, 
            how: str) -> None:
        """
        Fits the model to input and target DataFrames, given a normalizing method and a normalizing column.

        :param inputs: The inputs.
        :param target: The targets.
        :param normalizing_col: The column used to calculate the normalized target.
        :param how: The method used to normalize. Can be `subtract`, `divide`, or `multiply`.
        """

        if isinstance(targets, pd.Series):
            targets = pd.DataFrame(targets)
        target_col = list(targets.columns)[0]

        assert normalizing_col in inputs.columns, f"`normalizing_col` {normalizing_col} not found in `inputs`"
        assert (inputs.index == targets.index).all(), "index columns of inputs and targets DataFrames must be the same"

        normalizable_indices = list((inputs[inputs[normalizing_col].notna()]).index)
        normalizable_targets = targets.loc[normalizable_indices, target_col]
        normalizable_inputs = inputs.loc[normalizable_indices, :]

        if how == "subtract":
            normalizable_targets = normalizable_targets - normalizable_inputs[normalizing_col]
        elif how == "divide":
            if 0 in normalizable_inputs[normalizing_col]:
                if self.verbose != -1:
                    warnings.warn(f"`0` found in {normalizing_col} with `divide` normalization. \
                                Omitting from the normalizable rows.")
                normalizable_inputs = normalizable_inputs[normalizable_targets != 0]
                normalizable_targets = normalizable_targets[normalizable_targets != 0]
            normalizable_targets = normalizable_targets / normalizable_inputs[normalizing_col]
        elif how == "multiply":
            normalizable_targets = normalizable_targets * normalizable_inputs[normalizing_col]
        else:
            raise ValueError("Variable `how` can only be `subtract`, `divide`, or `multiply`.")

        if self.verbose != -1:
            print("[NormalizedModel] Fitting general model.")
        self.base_estimator.fit(inputs, targets[target_col])
        if len(normalizable_inputs) > 0:
            if self.verbose != -1:
                print("[NormalizedModel] Fitting normalized model.")
            self.normalizing_estimator.fit(normalizable_inputs, pd.Series(normalizable_targets))
        else:
            warnings.warn("[NormalizedModel] No normalizable columns found. Not fitting normalized model.")

        self.normalizing_col = normalizing_col
        self.normalize_how = how
    

    def predict(self, inputs: pd.DataFrame) -> np.array:
        """
        Predicts targets for a DataFrame of inputs.

        :param inputs: DataFrame of inputs, matching columns given in `fit`.

        :return preds: NumPy array of predictions.
        """

        if self.normalizing_col is None or self.normalize_how is None:
            raise ValueError("`fit` must be called successfully before predicting!")
        
        assert self.normalizing_col in inputs.columns, f"`normalizing_col` {self.normalizing_col} used in `fit` \
            must be in inputs given to `predict`"

        non_normalizable_inputs = inputs[inputs[self.normalizing_col].isna()]
        normalizable_inputs = inputs[inputs[self.normalizing_col].notna()]

        inputs["preds"] = np.nan

        if len(non_normalizable_inputs) > 0:
            base_predictions = self.base_estimator.predict(non_normalizable_inputs)
            inputs.loc[non_normalizable_inputs.index, "preds"] = base_predictions
        
        if len(normalizable_inputs) > 0:
            normalized_predictions = self.normalizing_estimator.predict(normalizable_inputs)
            if self.normalize_how == "subtract":
                normalized_predictions += normalizable_inputs[self.normalizing_col]
            elif self.normalize_how == "divide":
                normalized_predictions *= normalizable_inputs[self.normalizing_col]
            elif self.normalize_how == "multiply":
                normalized_predictions /= normalizable_inputs[self.normalizing_col]
            else:
                raise ValueError(f"The normalizing method is {self.normalize_how} but must be one of `subtract`, \
                                `divide`, or `multiply`. Please call `fit` again with one of these options.")
            inputs.loc[normalizable_inputs.index, "preds"] = normalized_predictions

        preds = inputs["preds"].to_numpy()
        inputs = inputs.drop(columns=["preds"])

        return preds