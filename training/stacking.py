import pandas as pd

from mlxtend.classifier import StackingCVClassifier
from sklearn.svm import SVC

from training.classifier import Classifier
from training.utils import tune_hyperparameters, save_models, load_models

data = pd.read_pickle("data/train.pkl")
y = data["winner"].copy()
X = data.drop("winner").copy()

seed = 0
meta_classifier = SVC()
lvl1_classifiers = load_models("lvl1")

stacking_paramaters = {"classifiers": lvl1_classifiers, "shuffle": False, "use_probas": True, 
                        "meta_classifier": meta_classifier, "n_jobs": -1}

stacking_hyperparameters = {"meta_classifier__kernel": ["linear", "rbf", "poly"],
          "meta_classifier__C": [1, 2],
          "meta_classifier__degree": [3, 4, 5],
          "meta_classifier__probability": [True]}

sclf = StackingCVClassifier(classifiers = lvl1_classifiers,
                            shuffle = False,
                            use_probas = True,
                            cv = 5,
                            meta_classifier = meta_classifier, 
                            random_state=0)

stacking = Classifier("stacking", StackingCVClassifier, stacking_paramaters, stacking_hyperparameters, seed)

stacking = tune_hyperparameters([stacking], X, y, "stacking")
save_models(stacking, "stacking")
