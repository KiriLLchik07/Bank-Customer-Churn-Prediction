from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from app.api.schemas import CustomerData, PredictionResponse
from src.predict_churn import CustomerChurnPredictor


predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager для управления жиненным циклом приложения
    - startup: загрузка модели при запуске
    - shutdown: очистка ресурсов при остановке
    """
    global predictor
    try:
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "models" / "catboost_tuned_20251010_190010.pkl"
        predictor = CustomerChurnPredictor(model_path=str(model_path))
        print("ML модель успешно загружена")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        predictor = None

    yield

    print("Приложение останавливается")

app = FastAPI(
    title="Bank Churn Prediction API",
    description="API для прогнозирования оттока клиентов банка",
    version='1.0.0',
    lifespan=lifespan
)

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get('/', response_model=HealthResponse)
async def root():
    """
    Корневой эндпоинт для проверки работы API
    """
    return {"status": "healthy", "message": "Bank Churn Prediction API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Предсказание оттока клиента

    - **customer**: Данные клиента для анализа
    """
    global predictor

    if predictor is None:
        raise HTTPException(status_code=500, detail="ML модель не загружена!")
    
    try:
        customer_dict = customer.model_dump()

        customer_dict.update({
            'Gender': 1 if customer_dict['Gender'] == 'Male' else 0,
            'Geo_Germany': 1 if customer_dict['Geography'] == "Germany" else 0,
            'Geo_France': 1 if customer_dict['Geography'] == 'France' else 0,
            'Geo_Spain': 1 if customer_dict['Geography'] == 'Spain' else 0,
            'HasCrCard': 1.0 if customer_dict['HasCrCard'] else 0.0,
            'IsActiveMember': 1.0 if customer_dict['IsActiveMember'] else 0.0,
        })

        customer_dict.pop('Geography', None)

        additional_features = {
            'Is_Senior_Active': 0,
            'Active_With_Multiple_Products': 0,
            'Value_Client': 0,
            'New_HighRisk': 0,
            'German_Female_Risk': 0,
            'AgeGroup_18-30': 0,
            'AgeGroup_31-40': 0,
            'AgeGroup_41-50': 0,
            'AgeGroup_51-60': 0,
            'AgeGroup_60+': 0
        }
        
        customer_dict.update(additional_features)
    
        result = predictor.predict_churn(customer_dict)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

# uvicorn main:app --reload
