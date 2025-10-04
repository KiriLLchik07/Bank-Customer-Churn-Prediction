import numpy as np
import pandas as pd

class ProprocessingData:
    def __init__(self, df):
        self.df = df.copy()
    
    def handle_missing_values(self):
        print("Пропуски до обработки:\n")
        print(self.df.isna().sum())
                
        categorical_col = 'Geography'
        mode_val = self.df[categorical_col].mode()[0]
        self.df[categorical_col] = self.df[categorical_col].fillna(mode_val)
        print(f"\nЗаполнены пропуски в {categorical_col} значением: {mode_val}")
        
        numerical_col = ['Age', 'HasCrCard', 'IsActiveMember']
        for col in numerical_col:
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_val)
            print(f"Заполнены пропуски в {col} значением: {median_val}")
        
        print("\nПропуски после обработки:")
        print(self.df.isnull().sum())
        
        return self.df

    def remove_technical_columns(self):
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        
        self.df = self.df.drop(columns=columns_to_drop)
        
        print(f"\nУдалены столбцы: {columns_to_drop}")
        print(f"Оставшееся количество признаков: {self.df.shape[1]}")
        
        return self.df

    def handle_outliers_robust(self):
        df_clean = self.df
        
        lower_bound = df_clean['Age'].quantile(0.01)
        upper_bound = df_clean['Age'].quantile(0.99)
        
        df_clean['Age'] = np.clip(df_clean['Age'], lower_bound, upper_bound)
        
        balance_upper = df_clean['Balance'].quantile(0.99)
        df_clean['Balance'] = np.clip(df_clean['Balance'], 0, balance_upper)

        self.df = df_clean
        
        return self.df


    def create_new_features(self):
        print("\n🎯 Создание новых признаков...")
        
        self.df['Is_Senior_Active'] = ((self.df['Age'] > 40) & 
                                    (self.df['IsActiveMember'] == 1)).astype(int)
        
        self.df['Active_With_Multiple_Products'] = ((self.df['IsActiveMember'] == 1) & 
                                                (self.df['NumOfProducts'] > 1)).astype(int)
        
        self.df['Value_Client'] = ((self.df['Balance'] > self.df['Balance'].median()) &
                                (self.df['NumOfProducts'] >= 2)).astype(int)
                
        self.df['New_HighRisk'] = ((self.df['Tenure'] < 2) & 
                                (self.df['NumOfProducts'] == 1)).astype(int)
        
        self.df['German_Female_Risk'] = ((self.df['Geography'] == 'Germany') & 
                                        (self.df['Gender'] == 'Female')).astype(int)
        
        self.df['AgeGroup'] = pd.cut(self.df['Age'], 
                                    bins=[0, 30, 40, 50, 60, 100],
                                    labels=['18-30', '31-40', '41-50', '51-60', '60+'])
        
        print("   Созданы: Is_Senior_Active, Active_With_Multiple_Products, Value_Client")
        print("   New_HighRisk, German_Female_Risk, AgeGroup")
        return self.df


    def check_new_features_correlation(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = self.df[numeric_cols].corr(method='spearman')
        
        new_features = ['Is_Senior_Active', 'New_HighRisk', 
                        'Active_With_Multiple_Products', 'Value_Client', 'German_Female_Risk']
        
        print("\n🔍 Корреляции новых признаков: (крит. значение считаем >= 0.6)")
        for new_feat in new_features:
            correlations = corr_matrix[new_feat].sort_values(ascending=False)
            high_corr = correlations[abs(correlations) >= 0.5]
            if len(high_corr) > 1:
                print(f"{new_feat}: {high_corr.to_dict()}")
            else:
                print(f"{new_feat}: ✅ Нет сильных корреляций")
        
        return corr_matrix

    def encode_categorical_features(self):
        geography_dummies = pd.get_dummies(self.df['Geography'], prefix='Geo').astype('int32')
        self.df = pd.concat([self.df, geography_dummies], axis=1)
        
        self.df['Gender'] = self.df['Gender'].map({"Female": 0, "Male": 1})
        
        age_dummies = pd.get_dummies(self.df['AgeGroup'], prefix='AgeGroup').astype('int32')
        self.df = pd.concat([self.df, age_dummies], axis=1)
        
        self.df = self.df.drop(['Geography', 'AgeGroup'], axis=1)
        
        print("\nЗакодированы категориальные признаки!")
        
        return self.df
    
    def preprocessing(self):
        self.handle_missing_values()
        self.remove_technical_columns()
        self.handle_outliers_robust()
        self.create_new_features()
        self.check_new_features_correlation()
        self.encode_categorical_features()
        print("\n✅ Предобработка завершена!")
        return self.df
