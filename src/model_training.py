from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report, precision_score, recall_score, roc_curve, auc
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class TrainModels:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions = {}
    
    def fit_models(self,):
        logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
        logistic_regression.fit(self.X_train, self.y_train)
        self.models['LogisticRegression'] = logistic_regression

        knn = KNeighborsClassifier()
        knn.fit(self.X_train, self.y_train)
        self.models['KNeighborsClassifier'] = knn

        decision_tree = DecisionTreeClassifier(random_state=42)
        decision_tree.fit(self.X_train, self.y_train)
        self.models['DecisionTreeClassifier'] = decision_tree

        random_forest = RandomForestClassifier(random_state=42)
        random_forest.fit(self.X_train, self.y_train)
        self.models['RandomForestClassifier'] = random_forest

        xgboost = XGBClassifier(random_state=42)
        xgboost.fit(self.X_train, self.y_train)
        self.models['XGBClassifier'] = xgboost
        
        lightgbm = LGBMClassifier(random_state=42, verbose=0)
        lightgbm.fit(self.X_train, self.y_train)
        self.models['LGBMClassifier'] = lightgbm

        catboost = CatBoostClassifier(random_state=42, verbose=0)
        catboost.fit(self.X_train, self.y_train)
        self.models['CatBoostClassifier'] = catboost

        return self.models
    
    def evaluate_models(self):
        for name_model, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            self.predictions[name_model] = {
                "roc_auc_score": roc_auc_score(self.y_test, y_pred_proba),
                "f1_score": f1_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),
                "classification_report": classification_report(self.y_test, y_pred)
            }

        return self.predictions
    
    def plot_roc_curve(self):

        for name_model, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:,1]
        
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2,
            label=f'{name_model} (AUC = {roc_auc:.3f})')

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for {name_model}')
            plt.legend()
            plt.grid(True)
            plt.show()
