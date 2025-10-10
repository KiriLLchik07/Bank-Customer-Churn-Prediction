from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os

class PrepareData:
    def __init__(self, df):
        self.df = df.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_balanced = None
        self.y_train_balanced = None
        self.scaler = None

    def splitting(self):
        X = self.df.drop(columns=['Exited'])
        y = self.df['Exited']        
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42, stratify=y
        )

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scaling(self):
        self.scaler = RobustScaler()

        numeric_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

        numeric_features = [col for col in self.X_train.columns if self.X_train[col].nunique() > 2 
                            & self.X_train[col].nunique() !=4]

        self.X_train[numeric_features] = self.scaler.fit_transform(self.X_train[numeric_features])

        self.X_test[numeric_features] = self.scaler.transform(self.X_test[numeric_features])

        return self.X_train, self.X_test
    
    def balancing_classes(self, random_state=42):
        balancing_method = RandomUnderSampler(random_state=random_state)

        self.X_train_balanced, self.y_train_balanced = balancing_method.fit_resample(self.X_train, self.y_train)

        return self.X_train_balanced, self.y_train_balanced
    
        
    def save_to_pickle(self, output_dir='../data/processed'):
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.X_train_balanced, f'{output_dir}/X_train.pkl')
        joblib.dump(self.X_test, f'{output_dir}/X_test.pkl')
        joblib.dump(self.y_train_balanced, f'{output_dir}/y_train.pkl')
        joblib.dump(self.y_test, f'{output_dir}/y_test.pkl')
        
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        joblib.dump(self.X_train.columns.tolist(), f'{output_dir}/feature_names.pkl')
        
        print(f"✅ Данные и объекты сохранены в {output_dir}")
    
    def preparing(self):

        self.splitting()
        self.scaling()
        self.balancing_classes()
        self.save_to_pickle()

        return self.X_train_balanced, self.X_test, self.y_train_balanced, self.y_test
