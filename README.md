# 🏦 Bank Customer Churn Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry--dependency%20management-orange.svg)](https://python-poetry.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-green.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/catboost-1.2.8-yellow.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/xgboost-3.0.5-red.svg)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/lightgbm-4.6.0-lightblue.svg)](https://lightgbm.readthedocs.io/)
[![Imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.14.0-yellow.svg)](https://imbalanced-learn.org/stable/)
[![Optuna](https://img.shields.io/badge/optuna-4.5.0-purple.svg)](https://optuna.org/)
[![SHAP](https://img.shields.io/badge/shap-0.48.0-success.svg)](https://shap.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/pandas-2.0.0-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7.0-blueviolet.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-0.13.2-blueviolet.svg)](https://seaborn.pydata.org/)

ML-проект по прогнозированию оттока клиентов банка с помощью машинного обучения. Система идентифицирует клиентов с высоким риском ухода и предоставляет персонализированные рекомендации по их удержанию.

## 🎯 Ценность для бизнеса

- **Снижение оттока** на 20-30% через превентивные меры
- **Персонализированные кампании** удержания для разных сегментов клиентов
- **Экономия бюджета** на привлечение новых клиентов
- **Увеличение LTV** Пожизненной ценности клиента

## 📊 Ключевые выводы

### 🔍 Топ-5 факторов оттока:

1. **1 продукт банка** → 51% риск vs 5% при 3+ продуктах
2. **Неактивность клиента** → 45% увеличение риска оттока
3. **Возраст 45+ лет** → 35% увеличение риска
4. **Гендер** → мужчины более склонны к оттоку
5. **Очень высокий баланс** → риск потери VIP-клиентов

### 👥 Портреты клиентов:

- **🚨 Высокий риск**: 1 продукт + неактивен + возраст 45+ + мужчина
- **🟢 Низкий риск**: 3+ продукта + активен + молодой возраст + женщина

## 🏆 Model Performance

| Модель                | ROC-AUC   | F1-Score  | Precision | Recall |
| --------------------- | --------- | --------- | --------- | ------ |
| **CatBoost (тюнинг)** | **0.872** | **0.635** | 0.617     | 0.654  |
| LightGBM (тюнинг)     | 0.863     | 0.589     | 0.602     | 0.576  |
| Random Forest         | 0.847     | 0.581     | 0.594     | 0.568  |

## 🚀 Быстрый старт

### Предварительные требования

- Python 3.11+
- Poetry

### Установка и запуск

```bash
# Клонирование репозитория
git clone https://github.com/your-username/Bank-Customer-Churn-Prediction.git
cd Bank-Customer-Churn-Prediction

# Установка зависимостей через Poetry
poetry install

# Активация виртуального окружения
poetry shell
```

## 📁 Project Structure

```
Bank-Customer-Churn-Prediction/
├── ⚙️ config/               # Конфигурационные файлы
│   ├── risk_factors.yaml    # Факторы риска
│   └── recommendations.yaml # Бизнес-рекомендации
├── 📊 data/                 # Исходные и обработанные данные
├── 🤖 models/               # Обученные модели
├── 📓 notebooks/            # Jupyter notebooks для анализа
│   ├── 01_primary_data_review.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_data_preparation.ipynb
│   ├── 05_modeling_and_experiments.ipynb
│   └── 06_model_interpretation.ipynb
├── 📈 reports/              # Отчеты и визуализации
├── 🔧 src/                  # Исходный код
│   ├── customer_generator.py     # Генератор тестовых клиентов
│   ├── data_preparation.py     # Подготовка данных к моделированию
│   ├── hyperparametr_config.py     # Сетка гиперпаараметров для различных моделей
│   ├── hyperparametr_tuner.py     # Подбор гиперпараметров с помощью optuna
│   ├── model_manager.py    # Сохранение и загрузка моделей
│   ├── model_training.py    # Обучение и оценка моделей
│   ├── predict_churn.py     # Основной класс для прогнозирования
│   └── preprocessing.py     # Предобработка данных
├── 🐳 app/                  # FastAPI и Streamlit приложения
│   ├── api/                 # FastAPI бэкенд
│   │   ├── main.py          # Основное приложение FastAPI
│   │   └── schemas.py       # Pydantic схемы данных
│   └── frontend/            # Streamlit фронтенд
│       ├── app.py           # Главное приложение Streamlit
│       ├── assets/          # Статические файлы
│       └── pages/           # Страницы приложения
├── 📄 README.md
└── 📋 pyproject.toml        # Зависимости проекта
```

## 💡 Бизнес-рекомендации

### 🎯 Приоритетные действия:

| Уровень риска  | Вероятность | Действия                                         |
| -------------- | ----------- | ------------------------------------------------ |
| 🚨 Критический | >60%        | Немедленное вмешательство, персональный менеджер |
| 🟡 Высокий     | 40-60%      | Приоритетное удержание, кросс-продажи            |
| 🟠 Средний     | 20-40%      | Активный мониторинг, email-кампании              |
| 🟢 Низкий      | <20%        | Стандартное обслуживание                         |

### Конкретные меры:

- **Клиентам с 1 продуктом**: предложить кредитную карту/сберегательный счет
- **Неактивным клиентам**: серия reactivation emails, персональный звонок
- **Клиентам 45+**: специальные программы, консультации по пенсионным накоплениям
- **Клиентам из Германии**: локализованные предложения

## 📈 Ключевые выводы из EDA

### Демография:

- **Дисбаланс классов**: 80% лояльные vs 20% ушедшие клиенты
- **Возраст**: Риск оттока растет после 45 лет
- **География**: Германия - самый проблемный регион (32% оттока)

### Поведенческие паттерны:

- **Количество продуктов**: Клиенты с 1 продуктом уходят в 3 раза чаще
- **Активность**: Неактивные клиенты в 2.5 раза чаще покидают банк
- **Баланс**: Клиенты с очень высокими балансами склонны к оттоку

## 🚀 API & Deployment

### FastAPI Backend

**Производственный REST API** для интеграции с другими системами и мобильными приложениями.

#### 🎯 Возможности:

- **RESTful API** с автоматической документацией Swagger
- **Валидация данных** через Pydantic схемы
- **Автодокументация** доступна по `/docs`
- **Готов к продакшн** использованию

#### 🚀 Запуск API:

```bash
cd app/api
poetry run uvicorn main:app --reload
```

#### 📚 Эндпоинты:

- `GET /` - Проверка здоровья API
- `POST /predict` - Предсказание оттока клиента

#### 🔧 Пример использования:

```python
import requests

customer_data = {
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 45,
    "Tenure": 3,
    "Balance": 120000.0,
    "NumOfProducts": 1,
    "HasCrCard": True,
    "IsActiveMember": False,
    "EstimatedSalary": 60000.0
}

response = requests.post("http://localhost:8000/predict", json=customer_data)
result = response.json()
```

### Streamlit Interface

**Интерактивный веб-интерфейс** для бизнес-пользователей и аналитиков.

#### 🎯 Возможности:

- **5 интерактивных страниц** с анализом данных
- **Визуализация SHAP** анализа и важности признаков
- **Real-time предсказания** с персонализированными рекомендациями
- **Профессиональный дизайн** с темной темой

#### 🚀 Запуск интерфейса:

```bash
cd app/frontend
poetry run streamlit run app.py
```

#### 📊 Страницы интерфейса:

1. **Главная страница** - Обзор проекта и ключевые метрики
2. **Разведочный анализ данных** - Интерактивный разведочный анализ данных
3. **Моделирование и эксперименты** - Сравнение алгоритмов и оптимизация гиперпараметров
4. **Интерпретация модели** - SHAP анализ и бизнес-инсайты
5. **Предсказать отток** - Real-time прогнозирование оттока клиентов

#### 🎨 Особенности интерфейса:

- **Адаптивный дизайн** для всех устройств
- **Интерактивные Plotly графики**
- **Цветовая схема** с золотыми акцентами
- **Профессиональные визуализации**

## 👥 Автор

- Кирилл Есаков - [KiriLLchik07](https://github.com/KiriLLchik07) | [Email](kirill3456777@mail.ru)

---

**⭐ Если этот проект был полезен, поставьте звезду на GitHub!**
