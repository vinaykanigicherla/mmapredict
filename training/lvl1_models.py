import argparse
from os.path import join
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.sparse import data
from sklearn.preprocessing import StandardScaler

# Classifier imports
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC, SVC

from training.utils import train_models, tune_hyperparameters, save_models
from training.classifier import Classifier


def main(data_path, seed):

    #Load Data
    data = pd.read_pickle(data_path)
    y = data["winner"].copy().apply(lambda winner: 0 if winner=="f1" else 1)
    X = data.drop("winner").copy()
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)


    #Make models to stack
    models = []
    Model = namedtuple("Model", ["name", "model", "parameter_grid"])

    models.append(Model("RandomForest", RandomForestClassifier, 
                        {"n_estimators": [200],
                        "class_weight": [None, "balanced"],
                        "max_features": ["auto", "sqrt", "log2"],
                        "max_depth" : [3, 4, 5, 6, 7, 8],
                        "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                        "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                        "criterion" :["gini", "entropy"]     ,
                        "n_jobs": [-1]}))

    models.append(Model("ExtraTrees", ExtraTreesClassifier, 
                        {"criterion" :["gini", "entropy"],
                        "splitter": ["best", "random"],
                        "class_weight": [None, "balanced"],
                        "max_features": ["auto", "sqrt", "log2"],
                        "max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                        "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                        "min_samples_leaf": [0.005, 0.01, 0.05, 0.10]}))

    models.append(Model("SVC", SVC, 
                        {"kernel": ["linear", "rbf", "poly"],
                        "gamma": ["auto"],
                        "C": [0.1, 0.5, 1, 5, 10, 50, 100],
                        "degree": [1, 2, 3, 4, 5, 6]}))

    models.append(Model("KNN", KNeighborsClassifier, 
                        {"n_neighbors": list(range(1,31)),
                        "p": [1, 2, 3, 4, 5],
                        "leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                        "n_jobs": [-1]}))

    models.append(Model("SGD", SGDClassifier, 
                        {"alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                        "penalty": ["l1", "l2"],
                        "n_jobs": [-1]}))

    models.append(Model("GBM", GradientBoostingClassifier,
                        {"learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001], 
                        "n_estimators": [200],
                        "max_depth": [2,3,4,5,6],
                        "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                        "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                        "max_features": ["auto", "sqrt", "log2"],
                        "subsample": [0.8, 0.9, 1]}))

    models.append(Model("MLP", MLPClassifier, 
                        {"hidden_layer_sizes": [(5), (10), (5,5), (10,10), (5,5,5), (10,10,10)],
                        "activation": ["identity", "logistic", "tanh", "relu"],
                        "learning_rate": ["constant", "invscaling", "adaptive"],
                        "max_iter": [100, 200, 300, 500, 1000, 2000],
                        "alpha": list(10.0 ** -np.arange(1, 10))}))

    models.append(Model("GNB", GaussianNB,
                        {"var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5]}))

    #Init, fit, tune, and save baseline and lvl1 classifiers
    classifiers = [Classifier(model.name, model.model, None, model.param_grid, seed) for model in models]
    baseline_classifiers = train_models(classifiers, X, y, "baseline")
    save_models(baseline_classifiers, "baseline")
    lvl1_classifiers = tune_hyperparameters(baseline_classifiers, X, y, "lvl1")
    save_models(lvl1_classifiers, "lvl1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to train data")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    data_path = join("..", args.data_path)
    main(data_path, args.seed)