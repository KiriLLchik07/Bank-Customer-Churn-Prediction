import streamlit as st

st.set_page_config(layout="wide", page_icon='🏡')

def load_css():
    with open("assets/styles/custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.markdown("""
<div class="title">
    <h1>🏦 Bank Customer Churn Prediction</h1>
</div>

<div class="about_project">
    ML-проект по прогнозированию оттока клиентов банка с помощью машинного обучения. 
    Система идентифицирует клиентов с высоким риском ухода и предоставляет персонализированные рекомендации по их удержанию.
</div>
""", unsafe_allow_html=True)

# Разделитель
st.markdown("---")

# Создаем две колонки для первого блока
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="section">
        <h2>Ценность для бизнеса</h2>
        <div class="card">
            <ul class="value-list">
                <li><strong>Снижение оттока</strong> на 20-30% через превентивные меры</li>
                <li><strong>Персонализированные кампании</strong> удержания для разных сегментов клиентов</li>
                <li><strong>Экономия бюджета</strong> на привлечение новых клиентов</li>
                <li><strong>Увеличение LTV</strong> Пожизненной ценности клиента</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="section">
        <h2>Model Performance</h2>
        <div class="card">
            <div class="table-container">
                <table class="performance-table">
                    <thead>
                        <tr>
                            <th>Модель</th>
                            <th>ROC-AUC</th>
                            <th>F1-Score</th>
                            <th>Precision</th>
                            <th>Recall</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr class="best-model">
                            <td><strong>CatBoost (тюнинг)</strong></td>
                            <td><strong>0.872</strong></td>
                            <td><strong>0.635</strong></td>
                            <td>0.617</td>
                            <td>0.654</td>
                        </tr>
                        <tr>
                            <td>LightGBM (тюнинг)</td>
                            <td>0.863</td>
                            <td>0.589</td>
                            <td>0.602</td>
                            <td>0.576</td>
                        </tr>
                        <tr>
                            <td>Random Forest</td>
                            <td>0.847</td>
                            <td>0.581</td>
                            <td>0.594</td>
                            <td>0.568</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Ключевые выводы
st.markdown("""
<div class="section">
    <h2>Ключевые выводы</h2>
    <div class="card">
        <h3>Топ-4 фактора оттока:</h3>
        <div class="factors-grid">
            <div class="factor-card">
                <div class="factor-number">1</div>
                <div class="factor-content">
                    <strong>1 продукт банка</strong><br>
                    → 51% риск vs 5% при 3+ продуктах
                </div>
            </div>
            <div class="factor-card">
                <div class="factor-number">2</div>
                <div class="factor-content">
                    <strong>Неактивность клиента</strong><br>
                    → 45% увеличение риска оттока
                </div>
            </div>
            <div class="factor-card">
                <div class="factor-number">3</div>
                <div class="factor-content">
                    <strong>Возраст 45+ лет</strong><br>
                    → 35% увеличение риска
                </div>
            </div>
            <div class="factor-card">
                <div class="factor-number">4</div>
                <div class="factor-content">
                    <strong>Очень высокий баланс</strong><br>
                    → риск потери VIP-клиентов
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Портреты клиентов - ОТДЕЛЬНЫМ БЛОКОМ
st.markdown("""
<div class="section">
    <div class="card">
        <h3>Портреты клиентов:</h3>
        <div class="customer-profiles">
            <div class="profile-high-risk">
                <span class="profile-icon">🚨</span>
                <div class="profile-content">
                    <strong>Высокий риск:</strong> 1 продукт + неактивен + возраст 45+ + мужчина
                </div>
            </div>
            <div class="profile-low-risk">
                <span class="profile-icon">🟢</span>
                <div class="profile-content">
                    <strong>Низкий риск:</strong> 3+ продукта + активен + молодой возраст + женщина
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <h2>Project Structure</h2>
    <div class="card">
        <div class="project-structure">
            <code>Bank-Customer-Churn-Prediction/<br>
├── config/               # Конфигурационные файлы<br>
│   ├── risk_factors.yaml    # Факторы риска<br>
│   └── recommendations.yaml # Бизнес-рекомендации<br>
├── data/                 # Исходные и обработанные данные<br>
├── models/               # Обученные модели<br>
├── notebooks/            # Jupyter notebooks для анализа<br>
│   ├── 01_primary_data_review.ipynb<br>
│   ├── 02_eda_analysis.ipynb<br>
│   ├── 03_preprocessing.ipynb<br>
│   ├── 04_data_preparation.ipynb<br>
│   ├── 05_modeling_and_experiments.ipynb<br>
│   └── 06_model_interpretation.ipynb<br>
├── reports/              # Отчеты и визуализации<br>
├── src/                  # Исходный код<br>
│   ├── customer_generator.py<br>
│   ├── data_preparation.py<br>
│   ├── hyperparametr_config.py<br>
│   ├── hyperparametr_tuner.py<br>
│   ├── model_manager.py<br>
│   ├── model_training.py<br>
│   ├── predict_churn.py<br>
│   └── preprocessing.py<br>
├── app/                  # FastAPI и Streamlit приложения<br>
├── README.md<br>
└── pyproject.toml        # Зависимости проекта</code>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Бизнес рекомендации
st.markdown("""
<div class="section">
    <h2>Бизнес рекомендации</h2>
    <div class="card">
        <h3>Приоритетные действия:</h3>
        <div class="table-container">
            <table class="risk-table">
                <thead>
                    <tr>
                        <th>Уровень риска</th>
                        <th>Вероятность</th>
                        <th>Действия</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="risk-critical">
                        <td>🚨 Критический</td>
                        <td>>60%</td>
                        <td>Немедленное вмешательство, персональный менеджер</td>
                    </tr>
                    <tr class="risk-high">
                        <td>🟡 Высокий</td>
                        <td>40-60%</td>
                        <td>Приоритетное удержание, кросс-продажи</td>
                    </tr>
                    <tr class="risk-medium">
                        <td>🟠 Средний</td>
                        <td>20-40%</td>
                        <td>Активный мониторинг, email-кампании</td>
                    </tr>
                    <tr class="risk-low">
                        <td>🟢 Низкий</td>
                        <td><20%</td>
                        <td>Стандартное обслуживание</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Конкретные меры - ОТДЕЛЬНЫМ БЛОКОМ
st.markdown("""
<div class="section">
    <div class="card">
        <h3 style="margin-top: 0;">Конкретные меры:</h3>
        <ul class="measures-list">
            <li><strong>Клиентам с 1 продуктом</strong>: предложить кредитную карту/сберегательный счет</li>
            <li><strong>Неактивным клиентам</strong>: выгодные скидки, персональная программа кэшбэка, персональный звонок</li>
            <li><strong>Клиентам 45+</strong>: специальные программы, консультации по пенсионным накоплениям</li>
            <li><strong>Клиентам из Германии</strong>: локализованные предложения</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Автор
st.markdown("""
<div class="section">
    <h2>Автор</h2>
    <div class="card author-card">
        <div class="author-info">
            <strong>Кирилл Есаков</strong><br>
            <div class="author-links">
                <a href="https://github.com/KiriLLchik07" target="_blank">GitHub: KiriLLchik07</a><br>
                <a href="mailto:kirill3456777@mail.ru">Email: kirill3456777@mail.ru</a>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <div class="github-star">
        <div class="card">
            <p style="text-align: center; margin: 0;">
                <strong>⭐ Если этот проект был полезен, поставьте звезду на GitHub!</strong>
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="footer">Bank Customer Churn Prediction © 2025</div>', unsafe_allow_html=True)
