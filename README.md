# 🏦 Bank Customer Churn Prediction

ML-проект по прогнозированию оттока клиентов банка с помощью машинного обучения.

_🚧 Проект в активной разработке. Скоро будет полное описание и результаты!_

**Датасет:** [Bank Customer Churn Prediction on Kaggle](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)

**Цель:** Прогнозировать, покинет ли клиент банк (отток), исходя из его демографических и финансовых характеристик.

---

## Требования

- Python 3.11+
- Poetry

## Установка и настройка

1. **Клонирование репозитория**

```bash
git clone https://github.com/your-username/Bank-Customer-Churn-Prediction.git
cd Bank-Customer-Churn-Prediction
```

## 🛠️ Настройка разработки

### Структура проекта

```
Bank-Customer-Churn-Prediction/
├── data/           # Файлы данных
├── notebooks/      # Jupyter notebooks для EDA и разработки
├── src/            # Исходный код Python
├── models/         # Обученные модели
├── reports/        # Аналитические отчеты и визуализации
└── pyproject.toml  # Зависимости проекта
```

### Использование Poetry

## 📑 Оглавление

- [⚡ Быстрый старт](#⚡быстрый-старт)
- [📝 Инструкция по работе с Poetry](#📝инструкция-по-работе-с-poetry)
  - [Установка Poetry](#установка-poetry📥)
  - [Инициализация проекта с Poetry](#инициализация-проекта-с-poetry⚙️)
  - [Настройка pyproject.toml](#создание-toml-файла-для-определения-версий-библиотек🛠️)
  - [Установка зависимостей](#установка-зависимостей⬇️)
  - [Работа с окружением](#активация-виртуального-окружения▶️)
  - [Полезные команды](#полезные-команды-poetry✨)

---

# 📝 Инструкция по работе с Poetry

## Установка Poetry📥

```bash
pip install poetry
```

## Инициализация проекта с Poetry⚙️

```bash
poetry init
```

## Создание .toml файла для определения версий библиотек🛠️

```toml
[tool.poetry]
name = "bank-customer-churn-prediction"
version = "0.1.0"
description = "A machine learning project to predict bank customer churn."
authors = ["your_Name <your_email>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
torch = "^2.0.0"
pandas = "^2.0.0"
numpy = "^1.24.0"
seaborn = "^0.13.0"
jupyter = "^1.0.0"
matplotlib = "^3.7.0"
scikit-learn = "^1.7.2"
optuna = "^4.5.0"
catboost = "^1.2.8"
lightgbm = "^4.6.0"
xgboost = "^3.0.5"
imbalanced-learn = "^0.14.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

```

## Установка зависимостей⬇️

```bash
poetry install
```

## Активация виртуального окружения▶️

### На время работы (НЕ ЗАБУДЬ выключить!)

```bash
poetry shell
```

### Запуск отдельных файлов с помощью poetry:

```bash
poetry run python your_script.py
```

## Деактивация виртуального окружения⛔

### Если активировали через shell

```bash
exit
```

### ИЛИ

```bash
deactivate
```

## Полезные команды poetry✨

### Обновить все зависимости

```bash
poetry update
```

### Показать дерево зависимостей

```bash
poetry show --tree
```

### Показать все установленные пакеты

```bash
poetry show
```

### Создать requirements.txt (если нужно)

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### Добавить пакет

```bash
poetry add package-name
```

### Удалить пакет

```bash
poetry remove package-name
```

### Проверить корректность pyproject.toml

```bash
poetry check
```

### Проверить активное окружение

```bash
poetry env info --path
```

### Список всех окружений Poetry

```bash
poetry env list
```

### Удалить окружение

```bash
poetry env remove python
```

# ⚡Быстрый старт

### 📥Установка poetry:

```bash
pip install poetry
```

### ⚙️Инициализация проекта с Poetry

```bash
poetry init
```

### 🛠️Создание .toml файла для определения версий библиотек

```toml
[tool.poetry]
name = "bank-customer-churn-prediction"
version = "0.1.0"
description = "A machine learning project to predict bank customer churn."
authors = ["your_Name <your_email>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
torch = "^2.0.0"
pandas = "^2.0.0"
numpy = "^1.24.0"
seaborn = "^0.13.0"
jupyter = "^1.0.0"
matplotlib = "^3.7.0"
scikit-learn = "^1.7.2"
optuna = "^4.5.0"
catboost = "^1.2.8"
lightgbm = "^4.6.0"
xgboost = "^3.0.5"
imbalanced-learn = "^0.14.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

```

### ⬇️Установка зависимостей

```bash
poetry install
```

### ▶️Активация виртуального окружения

```bash
poetry shell
```

### ⛔Деактивация виртуального окружения

```bash
exit
```

## 📦 Установка через pip

Если вы не используете Poetry:

```bash
pip install -r requirements.txt
```

## 📊 Обзор проекта

_Подробное описание проекта, выводы EDA и результаты моделирования будут добавлены по завершении._

---

## 📈 Запланированные функции

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Сравнение нескольких моделей ML
- Настройка гиперпараметров
- Оценка и интерпретация моделей
- Развертывание API (FastAPI/Streamlit)

---

```

```
