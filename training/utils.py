import pickle
import os
import pandas as pd
from tqdm import tqdm
from os.path import join

data_dir = "../data"
models_dir = "../models"

def save_models(classifiers: list, name: str):
    """Save the models in classifiers to dir in /models as .pkl files"""
    if not os.path.isdir(join(models_dir, name)): os.mkdir(join(models_dir, name))
    save_dir = join(models_dir, name)
    print(f"Saving {name} models to {save_dir}...")
    [pickle.dump(clf.classifier, open(join(save_dir, clf.classifier_name + ".pkl"), "wb")) for clf in tqdm(classifiers)]

def load_models(dir: str):
    """Load models from dir contained in /models"""
    print(f"Loading models from {dir} ...")
    return [pickle.load(open(path, "rb")) for path in os.listdir(join(models_dir, dir))]

def train_models(classifiers: list, X: pd.DataFrame, y: pd.DataFrame, name: str):
    """Train a list of classifiers"""
    print(f"Training {name} models...")
    return [clf.fit(X, y) for clf in tqdm(classifiers)] 

def tune_hyperparameters(classifiers: list, X: pd.DataFrame, y: pd.DataFrame, name: str):
    """Only tune hyperparameters of models in list of classifiers"""
    print(f"Tuning hyperaparameters for {name} models...")
    return [clf.tune_hyperparameters(X, y) for clf in tqdm(classifiers)]