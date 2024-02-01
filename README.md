# sklearn-special-ensembles :magic_wand:
A library that creates robust, special-purpose ensembles from `sklearn`-like base models (including `lightgbm`, `xgboost`, and `catboost`).

## Install with PyPi :white_check_mark:
```
pip install sklearn-special-ensembles
```

## Models and examples :rocket:
### NormalizedModel
`NormalizedModel` normalizes the target using another feature in the input `DataFrame`, either by subtracting it from the target, dividing the target by it, or multiplying the target by it. It trains one model on all available data and another on only those rows that are normalizable. During inference, non-normalizable rows are predicted by the general model, while normalizable rows are predicted by the normalized model.
```python
from sklearn_special_ensembles.models import NormalizedModel
from sklearn_special_ensembles.tests.generate_dummy_dataframe import generate_dummy_dataframe
from lightgbm import LGBMRegressor

train_df, test_df = generate_dummy_dataframe(num_categorical_predictors=2, categories_by_column=[[1, 2], [3, 4]])

base_model = LGBMRegressor(verbose=-1)
normalized_model = NormalizedModel(base_estimator=base_model)

normalized_model.fit(train_df.drop(columns=["target"]), train_df["target"], normalizing_col="numerical_0", how="divide")
preds = normalized_model.predict(test_df.drop(columns=["target"]))
```

### SegmentEnsemble
`SegmentEnsemble` fits a separate estimator to each subset of data corresponding to each tuple of unique values across some specified features (i.e., segments) in the `DataFrame`. This ensemble can also fit a general model to all available data and blend the segment-level predictions with the general predictions during inference.
```python
from sklearn_special_ensembles.tests.generate_dummy_dataframe import generate_dummy_dataframe
from sklearn_special_ensembles.models import SegmentEnsemble
from lightgbm import LGBMRegressor

train_df, test_df = generate_dummy_dataframe(num_categorical_predictors=2, categories_by_column=[[1, 2], [3, 4]])

base_model = LGBMRegressor(verbose=-1)
segment_ensemble = SegmentEnsemble(base_estimator=base_model)

segment_ensemble.fit(train_df.drop(columns=["target"]), train_df["target"], segment_cols=["categorical_0", "categorical_1"])
preds = segment_ensemble.predict(test_df.drop(columns=["target"]), percent_general_model=0.1)
```

### OutlierEnsemble
`OutlierEnsemble` is designed for working with data that has a designated `ID`-like column and some `ID`s that have a fundamentally different relationship with the predictors than the rest of the `ID`s in the dataset. It fits a separate model to these outlier `ID`s to preserve the signal in the rest of the data.
```python
from sklearn_special_ensembles.tests.generate_dummy_dataframe import generate_dummy_dataframe
from sklearn_special_ensembles.models import OutlierEnsemble
from lightgbm import LGBMRegressor

train_df, test_df = generate_dummy_dataframe(num_categorical_predictors=1, categories_by_column=[[1, 2, 3, 4]])

base_model = LGBMRegressor(verbose=-1)
outlier_ensemble = OutlierEnsemble(base_estimator=base_model)

outlier_ensemble.fit(train_df.drop(columns=["target"]), train_df["target"], id_col="categorical_0", outlier_ids=[1, 3])
preds = outlier_ensemble.predict(test_df.drop(columns=["target"]))
```

### FeatureSubsetEnsemble
`FeatureSubsetEnsemble` trains separate base learners on distinct subsets of the available features in the data. This technique adds powerful diversity to an ensemble and should be particularly helpful when working with noisy data and a large feature space.
```python
from sklearn_special_ensembles.tests.generate_dummy_dataframe import generate_dummy_dataframe
<<<<<<< Updated upstream
from sklearn_special_ensembles.models.FeatureSubsetEnsemble import FeatureSubsetEnsemble
=======
from sklearn_special_ensembles.models import FeatureSubsetEnsemble
from lightgbm import LGBMRegressor
>>>>>>> Stashed changes

train_df, test_df = generate_dummy_dataframe(num_categorical_predictors=2, categories_by_column=[[1, 2, 3, 4], [5, 6]])

base_model = LGBMRegressor(verbose=-1)
feature_ensemble = FeatureSubsetEnsemble(estimators=[base_model for _ in range(2)])

feature_ensemble.fit(
    train_df.drop(columns=["target"]),
    train_df["target"],
    train_col_groups=[["numerical_0", "numerical_1"],
                      ["categorical_0", "categorical_1"]]
)
preds = feature_ensemble.predict(test_df.drop(columns=["target"]))
```


### FoldableEnsemble
`FoldableEnsemble` trains separate base estimators on separate folds of the data and ensembles their predictions together during inference. The user can specify the indices of the folds used during training and the weights of the estimators used during inference.
```python
import copy
from sklearn_special_ensembles.tests.generate_dummy_dataframe import generate_dummy_dataframe
<<<<<<< Updated upstream
from sklearn_special_ensembles.models.FoldableEnsemble import FoldableEnsemble
=======
from sklearn_special_ensembles.models import FoldableEnsemble
from lightgbm import LGBMRegressor
>>>>>>> Stashed changes

train_df, test_df = generate_dummy_dataframe(num_categorical_predictors=1, categories_by_column=[[1, 2, 3, 4]])

base_model = LGBMRegressor(verbose=-1)
n_splits = 4
foldable_ensemble = FoldableEnsemble(estimators=[base_model for _ in range(n_splits)])

foldable_ensemble.fit(train_df.drop(columns=["target"]), train_df["target"])
preds = foldable_ensemble.predict(test_df.drop(columns=["target"]))
```

## Upcoming :soon:
Don't hesitate to reach out if you find any bugs in this package or want to contribute! In the meantime, I'll just be including more of my own ensembles as they become useful and some other Kaggler's ensembles as their details are made public.