MODEL_PARAMS_CONFIG = {
    'catboost': {
        'class': 'CatBoostClassifier',
        'fixed_params': {'random_state': 42, 'verbose': False, 'auto_class_weights': 'Balanced'},
        'grid_params': {
            'iterations': {'type': 'int', 'low': 200, 'high': 1000},
            'depth': {'type': 'int', 'low': 4, 'high': 8},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'l2_leaf_reg': {'type': 'float', 'low': 1, 'high': 10},
            'border_count': {'type': 'int', 'low': 32, 'high': 255},
            'random_strength': {'type': 'float', 'low': 0.1, 'high': 10},
            'bagging_temperature': {'type': 'float', 'low': 0.0, 'high': 1.0}
        }
    },
    'lightgbm': {
        'class': 'LGBMClassifier', 
        'fixed_params': {'random_state': 42, 'verbose': -1, 'class_weight': 'balanced'},
        'grid_params': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 8},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'num_leaves': {'type': 'int', 'low': 20, 'high': 100},
            'min_child_samples': {'type': 'int', 'low': 10, 'high': 100},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
            'reg_lambda': {'type': 'float', 'low': 0, 'high': 10}
        }
    },
    'xgboost': {
        'class': 'XGBClassifier',
        'fixed_params': {'random_state': 42},
        'grid_params': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 8},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bylevel': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
            'gamma': {'type': 'float', 'low': 0, 'high': 10},
            'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
            'reg_lambda': {'type': 'float', 'low': 0, 'high': 10}
        }
    }
}
