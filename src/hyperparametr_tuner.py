from datetime import datetime
import os
from catboost import CatBoostClassifier
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import optuna
from model_manager import ModelManager

class HyperparametrTuner:
    """
    Класс, позволяющий подобрать гиперпараметры модели с помощью **optuna**.
    """
    def __init__(self, X_train, y_train, params_config, 
                 n_trials=30, cv=5, random_state=42, scoring='roc_auc', direction='maximize'):
        """
        - **X_train**: train выборка;
        - **y_train**: тренировочные метки классов;
        - **params_config**: сетка гиперпараметров;
        - **n_trials(default=30)**: количество испытаний подбора гиперпараметров;
        - **cv(default=5)**: количество фолдов;
        - **random_state(default=42)**: сид генерации случайных чисел;
        - **scoring(default='roc_auc')**: метрика, которую будем оптимизировать;
        - **direction(default='maximize')**: направление оптимизации
        """
        self.X_train = X_train
        self.y_train = y_train
        self.params_config = params_config
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.scoring = scoring
        self.direction = direction
        self.best_params = {}
        self.results = {}
        self.model_manager = ModelManager()

        self.model_classes = {
            'CatBoostClassifier': CatBoostClassifier,
            'LGBMClassifier': LGBMClassifier,
            'XGBClassifier': XGBClassifier
        }

    def gererate_objective(self, model_name, model_config):
        """
        Генерация функции objective для **optuna**.

        ### Agrumetns:
            model_name: имя модели
            model_config: сетка гиперпараметров

        **return**: функцию objective для **optuna**
        """
        model_class = self.model_classes[model_config['class']]
        grid_params = model_config['grid_params']
        fixed_params = model_config['fixed_params']

        def objective(trial):
            params = {}
            for param_name, param_config in grid_params.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['values']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False),
                        step=param_config.get('step', None)
                    )

            params.update(fixed_params)

            model = model_class(**params)

            scores = cross_val_score(
                model, self.X_train, self.y_train, 
                scoring=self.scoring, cv=self.cv, n_jobs=-1
            )

            return scores.mean()
        
        return objective
    
    def tune_models(self, models_to_tune=None):
        """
        Подбор гиперпараметров с помощью **optuna**
        """
        if models_to_tune is None:
            models_to_tune = list(self.params_config.keys())

        print(f"🎯 Запуск подбора гиперпараметров для: {models_to_tune}")

        for model_name in models_to_tune:
            print(f"\n🔍 Оптимизация {model_name}...")
            objective = self.gererate_objective(model_name, self.params_config[model_name])
            
            study = optuna.create_study(
                direction=self.direction,
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            study.optimize(objective, n_trials=self.n_trials)
            self.best_params[model_name] = study.best_params
            self.results[model_name] = study

            print(f"✅ {model_name}: лучший {self.scoring} = {study.best_value:.4f}")
            print(f"   Лучшие параметры: {study.best_params}")

        return self.best_params
    
    def get_tuned_models(self):
        """
        Получение модифицированных моделей
        """
        tuned_models = {}
        
        for model_name, best_params in self.best_params.items():
            model_config = self.params_config[model_name]
            model_class = self.model_classes[model_config['class']]
            
            all_params = {**best_params, **model_config['fixed_params']}
            
            tuned_models[f"{model_name}_tuned"] = model_class(**all_params)
        
        return tuned_models

    def get_study_analysis(self, model_name):
        """
        Получение параметров для модели
        """
        if model_name in self.results:
            return self.results[model_name]
        else:
            print(f"Study для {model_name} не найден")
            return None
        
    def save_tuning_results(self, study_name="hyperparameter_tuning"):
        """
        Сохранение результатов подбора гиперпараметров
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{study_name}_{timestamp}.pkl"
        filepath = os.path.join(self.model_manager.models_dir, filename)
        
        results = {
            'best_params': self.best_params,
            'study_results': self.results,
            'config_used': self.params_config,
            'timestamp': timestamp
        }
        
        joblib.dump(results, filepath)
        print(f"✅ Результаты тюнинга сохранены: {filepath}")
        return filepath
    
    def load_tuning_results(self, filepath):
        """
        Загрузка тюнинга
        """
        results = joblib.load(filepath)
        self.best_params = results['best_params']
        self.results = results['study_results']
        print(f"✅ Результаты тюнинга загружены из: {filepath}")
        return results
