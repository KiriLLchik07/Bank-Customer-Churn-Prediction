from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    """
    Модель для данных клиента
    """
    CreditScore: int = Field(ge=300, le=850, description="Кредитный рейтинг (300-850)")
    Geography: str = Field(description="Страна", examples=["Germany", "France", "Spain"])
    Gender: str = Field(description="Пол", examples=["Female", "Male"])
    Age: int = Field(ge=18, le=90, description="Возраст клиента")
    Tenure: int = Field(ge=0, le=10, description="Время клиента в банке (лет)")
    Balance: float = Field(ge=0, description="Баланс на счете")
    NumOfProducts: int = Field(ge=1, le=4, description="Количество продуктов")
    HasCrCard: bool = Field(description="Наличие кредитной карты")
    IsActiveMember: bool = Field(description="Активен ли клиент")
    EstimatedSalary: float = Field(ge=0, description="Предполагаемая зарплата")

class PredictionResponse(BaseModel):
    """Модель для ответа с прогнозом"""
    success: bool
    churn_probability: float = Field(..., ge=0, le=1, description="Вероятность оттока")
    risk_level: str = Field(..., description="Уровень риска")
    recommended_action: str = Field(..., description="Рекомендуемое действие")
    risk_factors: list[str] = Field(..., description="Факторы риска")
    recommendations: list[str] = Field(..., description="Список рекомендаций")
    key_metrics: dict | None = Field(None, description="Ключевые метрики клиента")
