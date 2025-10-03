import numpy as np
import pandas as pd

def handle_missing_values(df):
    print("Пропуски до обработки:")
    print(df.isna().sum())
    
    df_clean = df.copy()
    
    categorical_col = 'Geography'
    if categorical_col in df_clean.columns and df_clean[categorical_col].isna().any():
        mode_val = df_clean[categorical_col].mode()[0]
        df_clean[categorical_col] = df_clean[categorical_col].fillna(mode_val)
        print(f"Заполнены пропуски в {categorical_col} значением: {mode_val}")
    
    numerical_col = 'Age'
    if numerical_col in df_clean.columns and df_clean[numerical_col].isnull().any():
        median_val = df_clean[numerical_col].median()
        df_clean[numerical_col] = df_clean[numerical_col].fillna(median_val)
        print(f"Заполнены пропуски в {numerical_col} значением: {median_val}")
    
    print("Пропуски после обработки:")
    print(df_clean.isnull().sum())
    
    return df_clean

def remove_technical_columns(df):
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    
    df_clean = df.drop(columns=columns_to_drop)
    
    print(f"Удалены столбцы: {columns_to_drop}")
    print(f"Оставшееся количество признаков: {df_clean.shape[1]}")
    
    return df_clean

def handle_outliers_robust(df):
    df_clean = df.copy()
    
    lower_bound = df_clean['Age'].quantile(0.01)
    upper_bound = df_clean['Age'].quantile(0.99)
    
    df_clean['Age'] = np.clip(df_clean['Age'], lower_bound, upper_bound)
    
    balance_upper = df_clean['Balance'].quantile(0.99)
    df_clean['Balance'] = np.clip(df_clean['Balance'], 0, balance_upper)
    
    return df_clean


def create_smart_features(df):
    df_engineered = df.copy()
    
    df_engineered['Age_Active_Interaction'] = df_engineered['Age'] * df_engineered['IsActiveMember']
    
    df_engineered['Is_Senior_Active'] = ((df_engineered['Age'] > 40) & 
                                        (df_engineered['IsActiveMember'] == 1)).astype(int)
    
    df_engineered['High_Risk_Group'] = ((df_engineered['NumOfProducts'] == 1) | 
                                       (df_engineered['Geography'] == 'Germany')).astype(int)
    
    df_engineered['Active_With_Multiple_Products'] = ((df_engineered['IsActiveMember'] == 1) & 
                                        (df_engineered['NumOfProducts'] > 1)).astype(int)
    
    df_engineered['Value_Client'] = ((df['Balance'] > df['Balance'].median()) &
                                (df['NumOfProducts'] >= 2)).astype(int)
    
    df_engineered['AgeGroup'] = pd.cut(df_engineered['Age'], 
                            bins=[0, 30, 40, 50, 60, 100],
                            labels=['18-30', '31-40', '41-50', '51-60', '60+'])

    return df_engineered


def check_new_features_correlation(df_engineered):
    
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df_engineered[numeric_cols].corr(method='spearman')
    
    new_features = ['Age_Active_Interaction', 'Is_Senior_Active', 'High_Risk_Group', 
                    'Active_With_Multiple_Products', 'Value_Client']
    
    print("🔍 Корреляции новых признаков:")
    for new_feat in new_features:
        correlations = corr_matrix[new_feat].sort_values(ascending=False)
        high_corr = correlations[abs(correlations) > 0.5]
        if len(high_corr) > 1:
            print(f"{new_feat}: {high_corr.to_dict()}")
        else:
            print(f"{new_feat}: ✅ Нет сильных корреляций")
    
    return corr_matrix


def encode_categorical_features(df):
    df_encoded = df.copy()
    
    geography_dummies = pd.get_dummies(df_encoded['Geography'], prefix='Geo')
    df_encoded = pd.concat([df_encoded, geography_dummies], axis=1)
    
    df_encoded['Gender'] = df_encoded['Gender'].map({"Female": 0, "Male": 1})
    
    age_dummies = pd.get_dummies(df_encoded['AgeGroup'], prefix='AgeGroup')
    df_encoded = pd.concat([df_encoded, age_dummies], axis=1)
    
    df_encoded = df_encoded.drop(['Geography', 'Gender', 'AgeGroup'], axis=1)
    
    print("Закодированы категориальные признаки")
    
    return df_encoded
