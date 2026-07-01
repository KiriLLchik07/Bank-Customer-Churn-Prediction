import streamlit as st
import requests

st.set_page_config(
    page_title="Предсказание оттока - Bank Churn",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"

def load_css():
    try:
        with open("assets/styles/eda.css", "r", encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
        with open("assets/styles/predictions.css", "r", encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
    except FileNotFoundError as e:
        st.warning(f"Файл стилей не найден: {e}")

load_css()

def check_api_health():
    """Проверка доступности API"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_churn(customer_data: dict) -> dict:
    """Отправка данных на FastAPI для предсказания"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=customer_data,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'success': False,
                'error': f"API ошибка: {response.status_code} - {response.text}"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'error': "Не удалось подключиться к API. Убедитесь, что FastAPI сервер запущен."
        }
    except requests.exceptions.Timeout:
        return {
            'success': False, 
            'error': "Таймаут запроса. Попробуйте еще раз."
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Неожиданная ошибка: {str(e)}"
        }

st.markdown("""
<div class="eda-container">
    <h1 class="eda-title">Предсказать отток клиента</h1>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-nav">
    <h3 style="margin-bottom: 1rem; border-bottom: 1px solid #2a2f38; padding-bottom: 0.5rem;">Статус системы</h3>
""", unsafe_allow_html=True)

api_status = check_api_health()
if api_status:
    st.sidebar.success("API активно")
else:
    st.sidebar.error("API недоступно")
    st.sidebar.info("""
    **Для запуска API выполните:**
    ```bash
    cd app/api
    poetry run uvicorn main:app --reload
    ```
    """)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-nav">
    <h3 style="margin-bottom: 1rem; border-bottom: 1px solid #2a2f38; padding-bottom: 0.5rem;">Данные клиента</h3>
""", unsafe_allow_html=True)

st.sidebar.subheader("Демография")
credit_score = st.sidebar.slider("Кредитный рейтинг", 300, 850, 650, 
                               help="Кредитный скоринг от 300 до 850")
geography = st.sidebar.selectbox("Страна", ["France", "Germany", "Spain"], 
                               help="Страна проживания клиента")
gender = st.sidebar.selectbox("Пол", ["Female", "Male"])
age = st.sidebar.slider("Возраст", 18, 100, 45)
tenure = st.sidebar.slider("Время в банке (лет)", 0, 10, 3)

st.sidebar.subheader("Финансы")
balance = st.sidebar.number_input("Баланс на счете (€)", 0, 250000, 120000, 
                                help="Текущий баланс на счете")
num_products = st.sidebar.selectbox("Количество продуктов", [1, 2, 3, 4],
                                  help="Количество банковских продуктов у клиента")
has_credit_card = st.sidebar.checkbox("Есть кредитная карта", True)
is_active = st.sidebar.checkbox("Активный клиент", False)
salary = st.sidebar.number_input("Предполагаемая зарплата (€)", 0, 200000, 60000)

predict_btn = st.sidebar.button("Предсказать риск оттока", 
                               type="primary", 
                               use_container_width=True,
                               disabled=not api_status)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    with st.spinner("Анализируем данные клиента с помощью ML модели..."):
        customer_data = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": float(balance),
            "NumOfProducts": num_products,
            "HasCrCard": has_credit_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": float(salary)
        }
        
        result = predict_churn(customer_data)
        
        if result['success']:
            st.markdown('<h2 class="eda-subtitle" style="margin:10px">Результаты анализа</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-metric">
                    <h3 style="color: #b8860b; margin-bottom: 1rem;">Вероятность оттока</h3>
                    <div style="font-size: 3rem; font-weight: bold; color: #ff4d4f;">
                        {result['churn_probability']:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="prediction-metric">
                    <h3 style="color: #b8860b; margin-bottom: 1rem;">Уровень риска</h3>
                    <div style="font-size: 2rem; font-weight: bold; color: #5294e2;">
                        {result['risk_level']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="prediction-metric">
                    <h3 style="color: #b8860b; margin-bottom: 1rem;">Рекомендация</h3>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #2ebd85;">
                        {result['recommended_action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            prob = result['churn_probability']
            if prob > 0.6:
                risk_class = "risk-critical"
                risk_text = "Критический риск - Требуется немедленное вмешательство"
            elif prob > 0.4:
                risk_class = "risk-high"
                risk_text = "Высокий риск - Приоритетное удержание"
            elif prob > 0.2:
                risk_class = "risk-medium"
                risk_text = "Средний риск - Активный мониторинг"
            else:
                risk_class = "risk-low"
                risk_text = "Низкий риск - Стандартное обслуживание"
            
            st.markdown(f'<div class="{risk_class}">{risk_text}</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="eda-card">
                <h3 style="color: #b8860b; margin-bottom: 1rem;">Факторы риска</h3>
            """, unsafe_allow_html=True)
            
            if result['risk_factors']:
                for factor in result['risk_factors']:
                    st.markdown(f'<div class="factor-item">• {factor}</div>', unsafe_allow_html=True)
            else:
                st.info("Нет значимых факторов риска")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="eda-card">
                <h3 style="color: #b8860b; margin-bottom: 1rem;">Рекомендации по удержанию</h3>
            """, unsafe_allow_html=True)
            
            for recommendation in result.get('recommendations', []):
                st.markdown(f'<div class="recommendation-item">• {recommendation}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if result.get('key_metrics'):
                st.markdown("""
                <div class="eda-card">
                    <h3 style="color: #b8860b; margin-bottom: 1rem;">Ключевые метрики клиента</h3>
                """, unsafe_allow_html=True)
                
                for metric, value in result['key_metrics'].items():
                    st.markdown(f'<div class="factor-item"><strong>{metric}:</strong> {value}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("Детали введенных данных"):
                st.json(customer_data)
                
        else:
            st.error(f"{result.get('error', 'Unknown error')}")

else:
    if not api_status:
        st.error("""
        ## FastAPI сервер недоступен
        
        **Для работы предсказаний необходимо запустить бэкенд:**
        
        ```bash
        # Откройте новый терминал и выполните:
        cd app/api
        poetry run uvicorn main:app --reload
        ```
        
        После запуска сервера обновите страницу.
        """)
    else:
        st.markdown("""
        <div class="customer-profile">
            <h3 style="color: #b8860b; text-align: center;">Введите данные клиента в боковой панели и нажмите "Предсказать риск оттока"</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="eda-subtitle" style="margin-bottom:10px;">Примеры типичных профилей</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="conclusion-card conclusion-high-risk">
            <h4>Клиент высокого риска</h4>
            <ul>
                <li>1 продукт банка</li>
                <li>Неактивный клиент</li>
                <li>Возраст 45+ лет</li>
                <li>Из Германии</li>
                <li>Мужчина</li>
            </ul>
            <p style="color: #ff4d4f; font-weight: bold; margin-top: 1rem;">Вероятность оттока: 27-42%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="conclusion-card conclusion-low-risk">
            <h4>Клиент низкого риска</h4>
            <ul>
                <li>3+ продукта банка</li>
                <li>Активный клиент</li>
                <li>Молодой возраст (18-35)</li>
                <li>Из Франции/Испании</li>
                <li>Женщина</li>
            </ul>
            <p style="color: #2ebd85; font-weight: bold; margin-top: 1rem;">Вероятность оттока: 1-5%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="eda-card">
        <h3 style="color: #b8860b;">Как работает предсказание?</h3>
        <div style="color: #a0a5b0;">
            <p>Предсказание выполняется с помощью <strong>CatBoost модели</strong> через <strong>FastAPI бэкенд</strong>:</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 1.4rem; font-weight: 700; color: #b8860b;">API</div>
                    <strong>FastAPI бэкенд</strong>
                    <p>REST API на Python</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.4rem; font-weight: 700; color: #b8860b;">ML</div>
                    <strong>CatBoost модель</strong>
                    <p>Градиентный бустинг</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.4rem; font-weight: 700; color: #b8860b;">AUC</div>
                    <strong>ROC-AUC: 0.872</strong>
                    <p>Высокая точность</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.4rem; font-weight: 700; color: #b8860b;">RT</div>
                    <strong>Real-time</strong>
                    <p>Мгновенные предсказания</p>
                </div>
            </div>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(37, 42, 50, 0.6); border-radius: 8px;">
                <strong>Архитектура системы:</strong>
                <p>Streamlit Frontend → FastAPI Backend → CatBoost Model → Predictions</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="text-align: center; color: #6c727d; margin: 0;">
        Предсказания выполняются через FastAPI бэкенд с использованием CatBoost модели
    </p>
</div>
""", unsafe_allow_html=True)
