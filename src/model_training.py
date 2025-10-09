from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report

class TrainModels:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.baseline_models = {}
        self.predictions = {}

    def fit_baseline_models(self):
        logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
        logistic_regression.fit(self.X_train, self.y_train)
        self.baseline_models['LogisticRegression'] = logistic_regression

        random_forest = RandomForestClassifier(random_state=42)
        random_forest.fit(self.X_train, self.y_train)
        self.baseline_models["RandomForestClassifier"] = random_forest

        return self.baseline_models

    def evaluate(self):
        for name_model, model in self.baseline_models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            self.predictions[name_model] = {
                "roc_auc_score": roc_auc_score(self.y_test, y_pred_proba),
                "f1_score": f1_score(self.y_test, y_pred),
                "classification_report": classification_report(self.y_test, y_pred)
            }

        return self.predictions


