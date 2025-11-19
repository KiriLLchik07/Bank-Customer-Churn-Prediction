from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report, precision_score, recall_score, roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model_manager import ModelManager
from datetime import datetime
import os

class TrainModels:
    """
    Класс, который позволяет обучить модели и сохраняют лучшую из них. Доступны следующие методы:

    - **fit_models()**: обучает базовые и продвинутые модели без подобранных гиперпараметров;
    - **add_tuned_models()**: добавляет к общему списку моделей наши продвинутые модели;
    - **evaluate_models()**: метод, предназначенный для оценки модели, с помощью метрик ML;
    - **compare_models_performance()**: cравнивает производительность всех моделей;
    - **get_best_model()**: выбирает наилучшую модель по указанной метрике;
    - **plot_roc_curve()**: отображает графики roc-auc кривых;
    - **optimize_classification_threshold()**: находит оптимальный порог классификации для выбранной метрики;
    - **evaluate_with_optimal_threshold()**: переоценка моделей с оптимальным порогом;
    - **save_model()**: сохраняет модель по указанному имени, с возможностью сохранения метрик;
    - **load_model_in_trainer()**: загружает модель в класс TrainModel для дальнейшего обучения;
    - **create_final_report()**: создает финальный отчет по модели;
    - **plot_confusion_matrix_final()**: отрисовывает confusion matrix для анализа ошибок;
    - **plot_feature_importance()**: визуализирует важность признаков;
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions = {}
        self.model_manager = ModelManager()
    
    def fit_models(self):
        """ 
        Обучает модели 
        
        **return**: словарь с обученными моделями
        """
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
        """
        Добавляет к общему списку моделей наши продвинутые модели

        **return**: обновленный словарь моделей
        """
        for name, model in tuned_models_dict.items():
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
        print(f"Добавлено {len(tuned_models_dict)} настроенных моделей")
        return self.models
    
    def evaluate_models(self):
        """
        Считает предсказания и метрики.

        **return**: словарь с метриками для каждой модели  
        """
        for name_model, model in self.models.items():
            y_pred_test = model.predict(self.X_test)
            y_pred_proba_test = model.predict_proba(self.X_test)[:, 1]

            y_pred_train = model.predict(self.X_train)
            y_pred_proba_train = model.predict_proba(self.X_train)[:, 1]
            self.predictions[name_model] = {
                "test_roc_auc": roc_auc_score(self.y_test, y_pred_proba_test),
                "test_f1_score": f1_score(self.y_test, y_pred_test),
                "test_precision": precision_score(self.y_test, y_pred_test),
                "test_recall": recall_score(self.y_test, y_pred_test),
                
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
        """
        Сравнивает производительность всех моделей.
        
        **return**: pd.Dataframe таблицу с результатами метрик всех моделей
        """
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
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test ROC-AUC', ascending=False)
        
        return comparison_df
    
    def get_best_model(self, metric='test_roc_auc'):
        """
        Выбирает наилучшую модель по указанной метрике.

        ### Argument:
            metric='test_roc_auc': метрика, по которой будет оцениваться модель

        **return**: возвращает название лучшей модели и результат по указанной метрике
        """
        best_model_name = None
        best_score = -1
        
        for model_name, metrics in self.predictions.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = model_name
        
        return best_model_name, best_score
    
    def plot_roc_curve(self):
        """ Отображает графики roc-auc кривых """
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

    def optimize_classification_threshold(self, model_name, metric='f1'):
        """
        Находит оптимальный порог классификации для выбранной метрики
        
        ### Arguments: 
            model_name: имя модели
            metric(default='f1'): метрика для которой будем подбирать оптимальный порог

        **return**: Возвращает наилучший порог классификации и наилучший score
        """
        
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(self.y_test, y_pred)
            elif metric == 'precision':
                score = precision_score(self.y_test, y_pred)
            elif metric == 'recall':
                score = recall_score(self.y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"Оптимальный порог для {model_name}: {best_threshold:.3f}")
        print(f"   {metric.capitalize()} с оптимальным порогом: {best_score:.4f}")
        
        return best_threshold, best_score
    
    def evaluate_with_optimal_threshold(self, model_name, threshold):
        """
        Переоценка модели с оптимальным порогом

        ### Arguments:

            model_name: название модели
            threshold: порог классификации

        **return**: словарь с результатами подсчитанных метрик
        """
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        y_pred_optimal = (y_pred_proba >= threshold).astype(int)
        
        optimal_metrics = {
            'f1_score': f1_score(self.y_test, y_pred_optimal),
            'precision': precision_score(self.y_test, y_pred_optimal),
            'recall': recall_score(self.y_test, y_pred_optimal),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        print(f"Метрики с порогом {threshold:.3f}:")
        for metric, value in optimal_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return optimal_metrics
    
    def save_model(self, model_name, metrics=None):
        """
        Сохраняет модель по указанному имени, с возможностью сохранения метрик.

        ### Arguments:

            model_name: название модели
            metrics(default=None): метрики, которые будем хранить вместе с моделью

        **return**: сохраненную модель
        """
        model = self.models[model_name]
        metadata = {
            'model_name': model,
            'training_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'performance_metrics': self.predictions.get(model_name, {}),
        }

        if metrics:
            metadata.update(metrics)

        return self.model_manager.save_model(model, model_name, metadata)
    
    def load_model_in_trainer(self, model_name_or_path, new_name=None):
        """
        Загружает модель в класс TrainModel для дальнейшего обучения

        ### Arguments:
            model_name_or_path: название модели или путь к модели
            new_name(default=None): новое имя модели

        **return**: модель
        """
        model = self.model_manager.load_model(model_name_or_path)

        if new_name:
            model_name = new_name
        else:
            model_name = os.path.basename(model_name_or_path).split('_')[0]

        self.models[model_name] = model
        print(f"Модель {model_name} загружена в trainer")
        return model

    def create_final_report(self, model_name, threshold=0.5):
        """
        Создаёт финальный отчёт по модели.

        ### Arguments: 

            model_name: название модели
            threshold(default=0.5): порог классификации

        **return**: словарь, содержащий метрики модели, а также порог классификации
        """
        model = self.models[model_name]
        
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        report = {
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'f1': f1_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'threshold': threshold
        }
        
        return report

    def plot_confusion_matrix_final(self, model_name, threshold=0.5):
        """
        Отрисовывает confusion matrix, для анализа ошибок.

        ### Arguments: 

            model_name: название модели
            threshold(default=0.5): порог классификации
        """        
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred, ax=ax, cmap='Blues')
        plt.title(f'Confusion Matrix ({model_name})\nThreshold = {threshold}')
        plt.show()
        
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        print("Бизнес-интерпретация:")
        print(f"   Правильно предсказали лояльных: {tn} клиентов")
        print(f"   Ложные срабатывания (напрасно беспокоили): {fp} клиентов") 
        print(f"   Пропущенные уходящие: {fn} клиентов")
        print(f"   Правильно найденные уходящие: {tp} клиентов")

    def plot_feature_importance(self, model_name, top_n=15):
        """
        Визуализация важности признаков

        ### Arguments: 

            model_name: название модели
            top_n(default=15): количество отображаемых признаков
        """
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X_train.columns if hasattr(self.X_train, 'columns') else [f'Feature_{i}' for i in range(len(importances))]
            
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Feature Importance - {model_name}')
            plt.barh(range(min(top_n, len(indices))), 
                    importances[indices][:top_n][::-1])
            plt.yticks(range(min(top_n, len(indices))), 
                    [feature_names[i] for i in indices[:top_n]][::-1])
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            print("Топ-10 самых важных признаков:")
            for i in range(min(10, len(indices))):
                print(f"   {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        else:
            print(f"Модель {model_name} не поддерживает feature importance")
