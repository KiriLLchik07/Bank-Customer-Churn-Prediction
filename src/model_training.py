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
    
    def add_tuned_models(self, tuned_models_dict):
        for name, model in tuned_models_dict.items():
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
        print(f"✅ Добавлено {len(tuned_models_dict)} настроенных моделей")
        return self.models
    
    def evaluate_models(self):
        for name_model, model in self.models.items():
            y_pred_test = model.predict(self.X_test)
            y_pred_proba_test = model.predict_proba(self.X_test)[:, 1]

            y_pred_train = model.predict(self.X_train)
            y_pred_proba_train = model.predict_proba(self.X_train)[:, 1]
            self.predictions[name_model] = {
                # Метрики на тестовых данных
                "test_roc_auc": roc_auc_score(self.y_test, y_pred_proba_test),
                "test_f1_score": f1_score(self.y_test, y_pred_test),
                "test_precision": precision_score(self.y_test, y_pred_test),
                "test_recall": recall_score(self.y_test, y_pred_test),
                
                # Метрики на тренировочных данных
                "train_roc_auc": roc_auc_score(self.y_train, y_pred_proba_train),
                "train_f1_score": f1_score(self.y_train, y_pred_train),
                "train_precision": precision_score(self.y_train, y_pred_train),
                "train_recall": recall_score(self.y_train, y_pred_train),

                "roc_auc_diff": roc_auc_score(self.y_train, y_pred_proba_train) - roc_auc_score(self.y_test, y_pred_proba_test),
                "f1_diff": f1_score(self.y_train, y_pred_train) - f1_score(self.y_test, y_pred_test),
                
                "classification_report": classification_report(self.y_test, y_pred_test)
            }

        return self.predictions
    
    def compare_models_performance(self):
        """Сравнивает производительность всех моделей"""
        comparison_data = []
        
        for model_name, metrics in self.predictions.items():
            comparison_data.append({
                'Model': model_name,
                'Test ROC-AUC': metrics['test_roc_auc'],
                'Test F1': metrics['test_f1_score'],
                'Train ROC-AUC': metrics['train_roc_auc'],
                'Train F1': metrics['train_f1_score'],
                'ROC-AUC Diff': metrics['roc_auc_diff'],
                'F1 Diff': metrics['f1_diff']
            })
        
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test ROC-AUC', ascending=False)
        
        return comparison_df
    
    def get_best_model(self, metric='test_roc_auc'):
        best_model_name = None
        best_score = -1
        
        for model_name, metrics in self.predictions.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = model_name
        
        return best_model_name, best_score
    
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
