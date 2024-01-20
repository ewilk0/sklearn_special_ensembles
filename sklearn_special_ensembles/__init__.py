import os, sys
sys.path.insert(0, os.path.abspath(".."))

from sklearn_special_ensembles.models import NormalizedModel
from sklearn_special_ensembles.models import FoldableEnsemble
from sklearn_special_ensembles.models import FeatureSubsetEnsemble

from sklearn_special_ensembles.tests import generate_dummy_dataframe