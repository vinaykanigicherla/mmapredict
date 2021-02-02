from sklearn.model_selection import GridSearchCV, KFold

class Classifier():
    def __init__(self, classifier_name, classifier, init_params, param_grid, seed):
        self.classifier_name = classifier_name
        self.seed = seed
        self.param_grid = param_grid

        #Init classifier
        self.init_params = init_params
        self.init_params["random_state"] = seed
        self.classifier = classifier(**self.init_params) if init_params else classifier(random_state=seed)

        #Dict to explicitly store best stats
        self.best_stats = {"best_params": None, "best_score": None}
    
    def fit(self, X, y):
        print(f"Fitting {self.classifier_name} model...")
        self.classifier.fit(X, y)

    def predict(self, X):
        self.classifier.predict(X)
    
    def tune_hyperparameters(self, X, y):
        print(f"Tuning hyperparameters for {self.classifier_name} model...")
        cv = KFold(n_splits=20, random_state=self.seed, shuffle=True)
        gscv = GridSearchCV(self.classifier, self.param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
        gscv.fit(X, y)
        self.classifier.set_parameters(**gscv.best_params_)
        self.best_stats["best_params"], self.best_stats["best_score"] = gscv.best_params_, gscv.best_score_
            