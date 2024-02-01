import copy
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm


class SegmentEnsemble():
    """
    An ensemble that trains different estimators on the unique values in a
    given `segment`, or column of the input data, and offers the ability to blend
    those predictions with a `general` model that is trained on all the data. 
    
    This ensemble could, e.g., be used to train separate models to predict consumer
    behavior in each county of a given region, learning the more intricate relationships
    between the predictors and the target for each segment.
    """


    def __init__(self, base_estimator, verbose: int = -1) -> None:
        """
        Initializes the class.

        :param base_estimator: The base estimator to be trained on the separate segments.
        :param verbose: -1 to suppress logs. Any other value to allow them.
        """

        self.base_estimator = copy.deepcopy(base_estimator)
        self.verbose = verbose

        self.segments_to_estimators = {}
        self.segment_cols = None


    def fit(self, 
            inputs: pd.DataFrame, 
            targets: pd.DataFrame,
            segment_cols: str,
            fit_general_model: bool = True,
            min_observations: int = None,
            segment_estimator_fit_args = {},
            general_estimator_fit_args = {}) -> None:
        """
        Fits the ensemble to the input and target DataFrames.

        :param inputs: The inputs.
        :param targets: The targets.
        :param segment_cols: The columns from which the unique value combinations will be derived.
        :param target_col: The name of the target column.
        :param fit_general_model: Whether or not to fit a general model to all the data.
        :param min_observations: The minimum (if any) number of observations required to train a
            model on that segment.
        :param segment_estimator_fit_args: Any args to pass to the segment estimator in fitting.
        :param general_estimator_fit_args: Any args to pass to the general estimator in fitting.
        """

        if isinstance(targets, pd.Series):
            target_col = targets.name
            targets = pd.DataFrame(targets)
        elif isinstance(targets, pd.DataFrame):
            target_col = targets.columns[0]
        else:
            raise ValueError("`targets` must be a pandas DataFrame or Series")

        assert len(inputs) == len(targets)

        df = pd.concat([inputs, targets], axis=1)
        grouped = df.groupby(segment_cols, observed=False)

        if self.verbose != -1:
            looper = tqdm(grouped)
            looper.set_description("[SegmentEnsemble] Fitting models to segments...")
        else:
            looper = grouped
        for name, group_df in looper:
            if min_observations is not None and len(group_df) < min_observations:
                if fit_general_model == False:
                    warnings.warn(f"Category {name} does not pass the minimum observation threshold so will not be fit. \
                                  It is dangerous to also opt to not fit a general model.")
                continue
            group_targets = group_df[target_col]
            group_df = group_df.drop(columns=segment_cols+[target_col])
            self.base_estimator.fit(group_df, group_targets, **segment_estimator_fit_args)
            self.segments_to_estimators[name] = copy.deepcopy(self.base_estimator)
        
        if fit_general_model:
            general_estimator = copy.deepcopy(self.base_estimator)
            general_estimator.fit(df.drop(columns=[target_col]), df[target_col], **general_estimator_fit_args)
            self.segments_to_estimators["general"] = copy.deepcopy(general_estimator)
        
        self.segment_cols = segment_cols
    

    def predict(self, 
                inputs: pd.DataFrame, 
                percent_general_model: float = 0., 
                segment_estimator_predict_args = {},
                general_estimator_predict_args = {}) -> np.array:
        """
        Predicts a given DataFrame of inputs.

        :param inputs: The inputs.
        :param percent_general_model: The percent of the general model's predictions to be blended
            with the segment-level models' predictions.
        :param segment_estimator_predict_args: Any args to be passed to the segment estimators
            during prediction.
        :param general_estimator_predict_args: Any args to be passed to the general estimator during
            prediction.
        """

        if self.segment_cols is None:
            raise ValueError(f"You must call `fit` successfully before predicting!")
        for col in self.segment_cols:
            if col not in inputs.columns:
                raise ValueError(f"Segment column {col} not in inputs")

        assert percent_general_model >= 0.

        inputs["preds"] = np.nan
        grouped = inputs.groupby(self.segment_cols, observed=False)

        for name, group_df in grouped:
            this_df = group_df.drop(columns=self.segment_cols+["preds"])
            this_key = name if name in self.segments_to_estimators else "general"
            this_estimator = self.segments_to_estimators[this_key]
            these_preds = this_estimator.predict(this_df, **segment_estimator_predict_args)
            inputs.loc[this_df.index, "preds"] = these_preds
        
        if percent_general_model > 0:
            percent_micro_model = 1 - percent_general_model
            general_model = self.segments_to_estimators["general"]
            overall_preds = general_model.predict(inputs.drop(columns=["preds"]), 
                                                  **general_estimator_predict_args)
            inputs["preds"] = (inputs["preds"]*percent_micro_model + 
                               overall_preds*percent_general_model)
        
        preds = inputs["preds"]
        inputs = inputs.drop(columns=["preds"])

        return preds