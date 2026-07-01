from pathlib import Path
from typing import Any, Dict
import pandas as pd
import joblib
import yaml
import operator

class CustomerChurnPredictor:
    def __init__(self, model_path: str = None):
        """
        Инициализация прогнозировщика с конфигурационными файлами
        """
        project_root = Path(__file__).parent.parent

        if model_path is None:
            model_path = project_root / "models" / "catboost_tuned_20251010_190010.pkl"

        self.model = joblib.load(model_path)

        self.risk_factors_config = self._load_config(project_root / "config" / "risk_factors.yaml")
        self.recommendations_config = self._load_config(project_root / "config" / "recommendations.yaml")

        self.operators = {
            '>': operator.gt,
            '>=': operator.ge,
            '<': operator.lt,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne
        }

    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурационного файла"""

        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def _check_condition(self, value, condition_value) -> bool:
        """Проверка условия с поддержкой операторов"""
        if value is None:
            return False
            
        if isinstance(condition_value, str):
            for op_symbol, op_func in self.operators.items():
                if condition_value.startswith(op_symbol):
                    try:
                        compare_value = float(condition_value[len(op_symbol):])
                        return op_func(value, compare_value)
                    except ValueError:
                        # Если не получается конвертировать в число, пробуем как строку
                        return op_func(str(value), condition_value[len(op_symbol):])
            
            # Если нет оператора, просто сравниваем как строки
            return str(value) == condition_value
        else:
            return value == condition_value

    
    def analyze_risk_factors(self, customer_data: dict, probability: float) -> list:
        """
        Универсальный анализ факторов риска на основе конфига
        """
        factors = []

        for feature, config in self.risk_factors_config.get('risk_factors', {}).items():
            value = customer_data.get(feature)

            for condition in config.get('conditions', []):
                if self._check_condition(value, condition['value']):
                    factors.append(condition['message'])
                    break
            
        return factors
    
    def generate_recommendations(self, customer_data: dict, probability: float) -> list:
        """
        Универсальная генерация рекомендаций на основе конфига
        """
        recommendations = []

        for feature, config in self.recommendations_config.get('recommendations', {}).items():
            if feature == 'probability':
                continue

            value = customer_data.get(feature)

            for condition in config.get('conditions', []):
                if self._check_condition(value, condition['value']):
                    recommendations.extend(condition.get('messages', []))

        prob_config = self.recommendations_config.get('recommendations', {}).get('probability', {})
        for condition in prob_config.get('conditions', []):
            if self._check_condition(probability, condition['value']):
                recommendations.extend(condition.get('messages', []))

        if not recommendations:
            default_config = self.recommendations_config.get('recommendations', {}).get('default', {})
            recommendations.extend(default_config.get('messages', []))
            
        return recommendations
    
    def predict_churn(self, customer_data: dict) -> Dict[str, Any]:
        """
        Основной метод для предсказания оттока
        """        
        test_data = pd.DataFrame([customer_data])
        
        probability = self.model.predict_proba(test_data)[0, 0]
        
        if probability > 0.6:
            risk_level = "Критический риск"
            action = "Немедленное вмешательство"
            color = "red"
        elif probability > 0.4:
            risk_level = "Высокий риск"
            action = "Приоритетное удержание" 
            color = "orange"
        elif probability > 0.2:
            risk_level = "Средний риск"
            action = "Активный мониторинг"
            color = "yellow"
        else:
            risk_level = "Низкий риск"
            action = "Стандартное обслуживание"
            color = "green"
        
        risk_factors = self.analyze_risk_factors(customer_data, probability)
        
        recommendations = self.generate_recommendations(customer_data, probability)
        
        return {
            'success': True,
            'churn_probability': round(probability, 4),
            'risk_level': risk_level,
            'color': color,
            'recommended_action': action,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'key_metrics': {
                'NumOfProducts': customer_data.get('NumOfProducts', 'N/A'),
                'IsActiveMember': customer_data.get('IsActiveMember', 'N/A'),
                'Age': customer_data.get('Age', 'N/A'),
                'Balance': customer_data.get('Balance', 'N/A')
            }
        }
