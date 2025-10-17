import random
import pandas as pd
from typing import Dict, List
import numpy as np

class CustomerGenerator:
    """Генератор реалистичных тестовых клиентов"""
    
    def __init__(self):
        self.distributions = {
            'CreditScore': {'min': 350, 'max': 850, 'mean': 650},
            'Age': {'min': 18, 'max': 92, 'mean': 38},
            'Tenure': {'min': 0, 'max': 10, 'mean': 5},
            'Balance': {'min': 0, 'max': 250000, 'mean': 76485},
            'EstimatedSalary': {'min': 11, 'max': 200000, 'mean': 100090},
        }
        
        self.probabilities = {
            'HasCrCard': 0.7,
            'IsActiveMember': 0.5,
            'Gender': {0: 0.45, 1: 0.55},  # Female, Male
            'NumOfProducts': {1: 0.5, 2: 0.35, 3: 0.1, 4: 0.05}
        }
    
    def generate_random_customer(self) -> Dict:
        """Генерация случайного клиента"""
        credit_score = int(np.random.normal(650, 100))
        credit_score = max(350, min(850, credit_score))
        
        age = int(np.random.normal(38, 10))
        age = max(18, min(92, age))
        
        tenure = random.randint(0, 10)
        balance = max(0, np.random.normal(76485, 50000))
        estimated_salary = max(0, np.random.normal(100000, 30000))
        
        gender = 1 if random.random() < self.probabilities['Gender'][1] else 0
        has_cr_card = 1.0 if random.random() < self.probabilities['HasCrCard'] else 0.0
        is_active_member = 1.0 if random.random() < self.probabilities['IsActiveMember'] else 0.0
        
        geo_choice = random.random()
        if geo_choice < 0.5:
            geography = {'Geo_France': 1, 'Geo_Germany': 0, 'Geo_Spain': 0}
        elif geo_choice < 0.75:
            geography = {'Geo_France': 0, 'Geo_Germany': 1, 'Geo_Spain': 0}
        else:
            geography = {'Geo_France': 0, 'Geo_Germany': 0, 'Geo_Spain': 1}
        
        products_choice = random.random()
        if products_choice < 0.5:
            num_products = 1
        elif products_choice < 0.85:
            num_products = 2
        elif products_choice < 0.95:
            num_products = 3
        else:
            num_products = 4
        
        age_groups = self._get_age_groups(age)
        
        customer = {
            'CreditScore': credit_score,
            'Gender': gender,
            'Age': float(age),
            'Tenure': tenure,
            'Balance': float(balance),
            'NumOfProducts': num_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': float(estimated_salary),
            'Is_Senior_Active': 1 if age > 40 and is_active_member == 1 else 0,
            'Active_With_Multiple_Products': 1 if is_active_member == 1 and num_products >= 2 else 0,
            'Value_Client': 1 if balance > 100000 else 0,
            'New_HighRisk': 1 if num_products == 1 and is_active_member == 0 else 0,
            'German_Female_Risk': 1 if geography['Geo_Germany'] == 1 and gender == 0 else 0,
            **geography,
            **age_groups
        }
        
        return customer
    
    def generate_customer_by_profile(self, profile: str) -> Dict:
        """Генерация клиента по заданному профилю"""
        base_customer = self.generate_random_customer()
        
        profile_modifications = {
            'high_risk': {
                'NumOfProducts': 1,
                'IsActiveMember': 0.0,
                'Age': random.randint(45, 70),
                'Geo_Germany': 1,
                'Geo_France': 0,
                'Geo_Spain': 0
            },
            'low_risk': {
                'NumOfProducts': 3,
                'IsActiveMember': 1.0,
                'Age': random.randint(18, 35),
                'Geo_France': 1,
                'Geo_Germany': 0,
                'Geo_Spain': 0
            },
            'premium_high_balance': {
                'Balance': random.randint(150000, 250000),
                'NumOfProducts': 2,
                'Age': random.randint(40, 65)
            },
            'young_inactive': {
                'Age': random.randint(18, 25),
                'IsActiveMember': 0.0,
                'NumOfProducts': 1
            }
        }
        
        if profile in profile_modifications:
            base_customer.update(profile_modifications[profile])
            base_customer.update(self._update_derived_features(base_customer))
        
        return base_customer
    
    def _get_age_groups(self, age: int) -> Dict:
        """Определение возрастных групп"""
        return {
            'AgeGroup_18-30': 1 if 18 <= age <= 30 else 0,
            'AgeGroup_31-40': 1 if 31 <= age <= 40 else 0,
            'AgeGroup_41-50': 1 if 41 <= age <= 50 else 0,
            'AgeGroup_51-60': 1 if 51 <= age <= 60 else 0,
            'AgeGroup_60+': 1 if age > 60 else 0
        }
    
    def _update_derived_features(self, customer: Dict) -> Dict:
        """Обновление производных признаков"""
        return {
            'Is_Senior_Active': 1 if customer['Age'] > 40 and customer['IsActiveMember'] == 1 else 0,
            'Active_With_Multiple_Products': 1 if customer['IsActiveMember'] == 1 and customer['NumOfProducts'] >= 2 else 0,
            'Value_Client': 1 if customer['Balance'] > 100000 else 0,
            'New_HighRisk': 1 if customer['NumOfProducts'] == 1 and customer['IsActiveMember'] == 0 else 0,
            'German_Female_Risk': 1 if customer.get('Geo_Germany', 0) == 1 and customer['Gender'] == 0 else 0,
        }
    
    def generate_batch(self, n_customers: int = 10, profile: str = None) -> List[Dict]:
        """Генерация батча клиентов"""
        customers = []
        for _ in range(n_customers):
            if profile:
                customer = self.generate_customer_by_profile(profile)
            else:
                customer = self.generate_random_customer()
            customers.append(customer)
        return customers
