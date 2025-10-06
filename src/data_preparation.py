from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

class PrepareData:
    def __init__(self, df):
        self.df = df.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_balanced = None
        self.y_train_balanced = None

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
        scaling = RobustScaler()

        numeric_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

        numeric_features = [col for col in self.X_train.columns if self.X_train[col].nunique() > 2 
                            & self.X_train[col].nunique() !=4]

        self.X_train[numeric_features] = scaling.fit_transform(self.X_train[numeric_features])

        self.X_test[numeric_features] = scaling.transform(self.X_test[numeric_features])

        return self.X_train, self.X_test
    
    def balancing_classes(self):
        smote = SMOTE(random_state=42)

        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)

        return self.X_train_balanced, self.y_train_balanced
    
    def preparing(self):

        self.splitting()
        self.scaling()
        self.balancing_classes()

        return self.X_train_balanced, self.X_test, self.y_train_balanced, self.y_test
