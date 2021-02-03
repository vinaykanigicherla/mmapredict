import argparse
import pandas as pd
from os.path import join

from mlxtend.classifier import StackingCVClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from training.classifier import Classifier
from training.utils import tune_hyperparameters, save_models, load_models

def main(data_path, seed):
    
    data = pd.read_pickle(data_path)
    y = data["winner"].copy().apply(lambda winner: 0 if winner=="f1" else 1)

    X = data.drop("winner").copy()
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    seed = 0
    meta_classifier = SVC()
    lvl1_classifiers = load_models("lvl1")

    stacking_paramaters = {"classifiers": lvl1_classifiers, "shuffle": False, "use_probas": True, 
                            "meta_classifier": meta_classifier, "n_jobs": -1}

    stacking_hyperparameters = {"meta_classifier__kernel": ["linear", "rbf", "poly"],
            "meta_classifier__C": [1, 2],
            "meta_classifier__degree": [3, 4, 5],
            "meta_classifier__probability": [True]}

    stacking = Classifier("stacking", StackingCVClassifier, stacking_paramaters, stacking_hyperparameters, seed)

    stacking = tune_hyperparameters([stacking], X, y, "stacking")
    save_models(stacking, "stacking")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to train data")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    data_path = join("..", args.data_path)
    main(data_path, args.seed)