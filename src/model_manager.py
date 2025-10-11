import os
from datetime import datetime
import joblib
import json

class ModelManager:
    def __init__(self, models_dir='../models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def save_model(self, model, model_name, metadata=None):
        """Сохраняет модель и метаданные"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        metadata_filename = f"{model_name}_{timestamp}_metadata.json"

        model_path = os.path.join(self.models_dir, model_filename)
        joblib.dump(model, model_path)

        if metadata is None:
            metadata = {}

        metadata.update({
            'model_name': model_name,
            'saved_at': timestamp,
            'model_file': model_filename
        })

        metadata_path = os.path.join(self.models_dir, metadata_filename)

        with open(metadata_path, 'w') as file:
            json.dump(metadata, file, indent=2)

    def load_model(self, model_name_or_path):
        """Загружает модель по имени или пути"""
        model_files = [f for f in os.listdir(self.models_dir) 
                        if f.startswith(model_name_or_path) and f.endswith('.pkl')]
            
        model_files.sort(reverse=True)
        model_path = os.path.join(self.models_dir, model_files[0])
        
        model = joblib.load(model_path)
        print(f"✅ Модель загружена: {model_path}")
        return model
    
    def get_model_metadata(self, model_name):
        """Возвращает метаданные модели"""
        metadata_files = [f for f in os.listdir(self.models_dir) 
                         if f.startswith(model_name) and f.endswith('_metadata.json')]
        
        metadata_files.sort(reverse=True)
        metadata_path = os.path.join(self.models_dir, metadata_files[0])
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
