import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Интерпретация модели - Bank Churn", 
    layout="wide"
)

def load_css():
    try:
        with open("assets/styles/eda.css", "r", encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Файл стилей не найден")

load_css()

st.markdown("""
<div class="eda-container">
    <h1 class="eda-title">Интерпретация модели</h1>
    <div class="eda-card">
        <p style="text-align: center; font-size: 1.2rem; margin: 0;">
            SHAP анализ, бизнес-инсайты и портреты клиентов для понимания факторов оттока.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-nav">
    <h3 style="margin-bottom: 1rem; border-bottom: 1px solid #2a2f38; padding-bottom: 0.5rem;">Навигация по интерпретации</h3>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "Выберите раздел:",
    ["SHAP анализ", "Beeswarm график", "Важность признаков", 
     "Waterfall анализ", "Портреты клиентов", "Бизнес-рекомендации"],
    label_visibility="collapsed"
)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

@st.cache_data
def load_interpretation_data():
    shap_importance = {
        'Признак': ['NumOfProducts', 'IsActiveMember', 'Age', 'Balance', 'Gender', 
                   'Geography_Germany', 'CreditScore', 'Tenure', 'Geography_Spain', 'EstimatedSalary'],
        'SHAP_значение': [1.85, 0.70, 0.34, 0.34, 0.22, 0.18, 0.15, 0.12, 0.08, 0.05],
        'Влияние': ['Очень сильное', 'Сильное', 'Умеренное', 'Умеренное', 'Умеренное',
                   'Слабое', 'Слабое', 'Слабое', 'Слабое', 'Очень слабое']
    }
    
    factors_impact = {
        'Фактор': ['1 продукт банка', 'Неактивность', 'Возраст 45+', 'Мужской пол', 
                  'Высокий баланс', 'Германия', 'Низкий кредитный рейтинг'],
        'Влияние_на_отток': ['Сильно увеличивает', 'Сильно увеличивает', 'Умеренно увеличивает',
                           'Умеренно увеличивает', 'Умеренно увеличивает', 'Слабо увеличивает', 'Слабо увеличивает'],
        'Бизнес_значение': ['Ключевой фактор риска', 'Поведенческий индикатор', 'Демографический риск',
                          'Гендерная предрасположенность', 'Риск потери VIP', 'Региональная особенность', 'Кредитный риск']
    }
    
    customer_profiles = {
        'Профиль': ['Высокий риск', 'Низкий риск', 'VIP клиент', 'Молодой неактивный'],
        'Вероятность_оттока': [27.3, 5.1, 1.2, 15.2],
        'Уровень_риска': ['Критический', 'Низкий', 'Низкий', 'Средний'],
        'Продукты': [1, 3, 4, 1],
        'Активность': ['Неактивен', 'Активен', 'Активен', 'Неактивен'],
        'Возраст': [52, 28, 45, 24]
    }
    
    return pd.DataFrame(shap_importance), pd.DataFrame(factors_impact), pd.DataFrame(customer_profiles)

shap_df, factors_df, profiles_df = load_interpretation_data()

if section == "SHAP анализ":
    st.markdown('<h2 class="eda-subtitle">SHAP анализ модели</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Что такое SHAP анализ?</h3>
            <div>
                <p><strong>SHAP (SHapley Additive exPlanations)</strong> - метод объяснения предсказаний машинного обучения.</p>
                <p>Показывает вклад каждого признака в итоговое предсказание для каждого отдельного клиента.</p>
            </div>
        </div>
                    
        <div class="eda-card">
            <div style="margin-top: 1rem;">
                <strong>Как интерпретировать:</strong>
                <ul>
                    <li><span style="color: #ff4d4f;">Положительные значения</span> - увеличивают вероятность оттока</li>
                    <li><span style="color: #2ebd85;">Отрицательные значения</span> - уменьшают вероятность оттока</li>
                    <li><strong>Высота столбца</strong> - сила влияния признака</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Ключевые инсайты</h3>
            <div>
                <p><strong>Топ-3 самых влиятельных признака:</strong></p>
                <ol>
                    <li><strong>Количество продуктов</strong> - определяющий фактор лояльности</li>
                    <li><strong>Активность клиента</strong> - ключевой поведенческий индикатор</li>
                    <li><strong>Возраст</strong> - важная демографическая характеристика</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="eda-card"><h4>Рейтинг важности признаков</h4>', unsafe_allow_html=True)
        
        fig = px.bar(shap_df.head(8), x='SHAP_значение', y='Признак',
                    orientation='h',
                    color='SHAP_значение',
                    color_continuous_scale='Viridis',
                    title='Топ-8 самых важных признаков по SHAP')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a5b0',
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif section == "Beeswarm график":
    st.markdown('<h2 class="eda-subtitle">Beeswarm график SHAP значений</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="eda-card">
        <h3 style="color: #b8860b;">Визуализация влияния признаков</h3>
        <div>
            <p><strong>Beeswarm график</strong> показывает распределение SHAP значений для всех клиентов в тестовой выборке.</p>
            <h3>Как читать график:</h3>
            <ul>
                <li><strong>По вертикали</strong> - признаки по важности</li>
                <li><strong>По горизонтали</strong> - влияние на прогноз</li>
                <li><strong style="color: #ff4d4f;">Красный цвет</strong> - высокие значения признака</li>
                <li><strong style="color: #5294e2;">Синий цвет</strong> - низкие значения признака</li>
                <li>Красные точки справа → высокий риск</li>
                <li>Синие точки слева → низкий риск</li>
                <li>Ширина облака → сила влияния</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-card insight-strong">
            <h4>Факторы, увеличивающие риск оттока</h4>
            <ul>
                <li><strong>1 продукт банка</strong> (красные точки справа)</li>
                <li><strong>Неактивность</strong> (синие точки справа)</li>
                <li><strong>Возраст 45+</strong> (красные точки справа)</li>
                <li><strong>Высокий баланс</strong> (красные точки справа)</li>
                <li><strong>Мужской пол</strong> (красные точки справа)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card insight-weak">
            <h4>Факторы, уменьшающие риск оттока</h4>
            <ul>
                <li><strong>3+ продукта</strong> (синие точки слева)</li>
                <li><strong>Активность</strong> (красные точки слева)</li>
                <li><strong>Молодой возраст</strong> (синие точки слева)</li>
                <li><strong>Средний баланс</strong> (синие точки слева)</li>
                <li><strong>Женский пол</strong> (синие точки слева)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="plotly-chart-container">
        <div style="text-align: center; padding: 1rem;">
            <h4>Beeswarm график SHAP значений</h4>
            <p><em>Визуализация влияния признаков на прогноз оттока для всех клиентов</em></p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.image("../../reports/beeswarm.png", 
            caption="Beeswarm график SHAP значений: красный - высокие значения признака увеличивают риск оттока, синий - низкие значения уменьшают риск")

elif section == "Важность признаков":
    st.markdown('<h2 class="eda-subtitle">Детальный анализ важности признаков</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="eda-card"><h4>Глобальная важность признаков</h4>', unsafe_allow_html=True)
        
        fig = px.pie(shap_df, values='SHAP_значение', names='Признак',
                    title='Распределение влияния признаков',
                    color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a5b0'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Бизнес-интерпретация</h3>
            <div style="color: #a0a5b0;">
                <div style="margin-bottom: 1rem;">
                    <strong>Топ-5 влиятельных признаков:</strong>
                    <ol>
                        <li><strong>NumOfProducts (28%)</strong> - Количество продуктов</li>
                        <li><strong>IsActiveMember (11%)</strong> - Активность клиента</li>
                        <li><strong>Age (5%)</strong> - Возраст</li>
                        <li><strong>Balance (5%)</strong> - Баланс счета</li>
                        <li><strong>Gender (3%)</strong> - Пол клиента</li>
                    </ol>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="eda-card"><h4>Направление влияния факторов</h4>', unsafe_allow_html=True)
    st.dataframe(factors_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Waterfall анализ":
    st.markdown('<h2 class="eda-subtitle">Waterfall анализ отдельных клиентов</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="eda-card">
        <h3 style="color: #b8860b;">Анализ индивидуальных предсказаний</h3>
        <div>
            <p><strong>Waterfall график</strong> показывает, как каждый признак повлиял на конкретное предсказание для отдельного клиента.</p>
            <p>Начинается с базового значения (средний прогноз по всем клиентам) и показывает вклад каждого признака.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="conclusion-card conclusion-high-risk">
            <h4>Клиент #1 - Высокий риск</h4>
            <div>
                <p><strong>Итог: f(x) = 4.636</strong></p>
                <p><strong>Факторы риска:</strong></p>
                <ul>
                    <li>2 продукта (+1.2)</li>
                    <li>Высокий баланс (+0.8)</li>
                    <li>Неактивен (+0.7)</li>
                    <li>Возраст 36 лет (+0.4)</li>
                </ul>
                <p><strong>Вероятность оттока: 42%</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="conclusion-card conclusion-low-risk">
            <h4>Клиент #2 - Низкий риск</h4>
            <div>
                <p><strong>Итог: f(x) = -1.667</strong></p>
                <p><strong>Защитные факторы:</strong></p>
                <ul>
                    <li>Активен (-0.9)</li>
                    <li>1 продукт (-0.6)</li>
                    <li>Германия (-0.3)</li>
                    <li>Мужчина (-0.2)</li>
                </ul>
                <p><strong>Вероятность оттока: 5%</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="conclusion-card conclusion-statistical">
            <h4>Клиент #3 - Средний риск</h4>
            <div>
                <p><strong>Итог: f(x) = 0.234</strong></p>
                <p><strong>Смешанные факторы:</strong></p>
                <ul>
                    <li>1 продукт (+0.8)</li>
                    <li>Возраст 41 год (+0.4)</li>
                    <li>Неактивна (+0.3)</li>
                    <li>Германия (-0.5)</li>
                    <li>Высокий баланс (-0.4)</li>
                </ul>
                <p><strong>Вероятность оттока: 15%</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="plotly-chart-container">
        <div style="text-align: center; padding: 1rem;">
            <h4>Waterfall анализ отдельных клиентов</h4>
            <p><em>Детальный разбор вклада каждого признака в конкретные предсказания</em></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Клиент высокого риска", "Клиент низкого риска", "Клиент среднего риска"])

    with tab1:
        st.subheader("Клиент #1 - Высокий риск оттока")
        st.image("../../reports/waterfall_1.png", 
                caption="Waterfall график для клиента с высоким риском оттока (42%)")
            
    with tab2:
        st.subheader("Клиент #2 - Низкий риск оттока") 
        st.image("../../reports/waterfall_0.png",
                caption="Waterfall график для клиента с низким риском оттока (5%)")
            
    with tab3:
        st.subheader("Клиент #3 - Средний риск оттока")
        st.image("../../reports/waterfall_2.png",
                caption="Waterfall график для клиента со средним риском оттока (15%)")

elif section == "Портреты клиентов":
    st.markdown('<h2 class="eda-subtitle">Портреты клиентов по уровням риска</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="eda-card"><h4>Сравнение профилей клиентов</h4>', unsafe_allow_html=True)
    
    fig = px.bar(profiles_df, x='Профиль', y='Вероятность_оттока',
                color='Уровень_риска',
                color_discrete_map={
                    'Критический': '#ff4d4f',
                    'Средний': '#ffc107',
                    'Низкий': '#2ebd85'
                },
                title='Вероятность оттока по профилям клиентов')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#a0a5b0'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="conclusion-card conclusion-high-risk">
            <h4>Клиент высокого риска</h4>
            <div>
                <p><strong>Характеристики:</strong></p>
                <ul>
                    <li><strong>1 продукт банка</strong></li>
                    <li><strong>Неактивен</strong> (IsActiveMember=0)</li>
                    <li><strong>Возраст 45+ лет</strong></li>
                    <li><strong>Мужской пол</strong></li>
                    <li><strong>Нулевой или очень высокий баланс</strong></li>
                </ul>
                <p><strong>Вероятность оттока: 27-42%</strong></p>
                <p><strong>Бизнес-значение:</strong> Критическая группа, требует немедленного вмешательства</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="conclusion-card conclusion-statistical" style="margin-top: 15px">
            <h4>Молодой неактивный клиент</h4>
            <div>
                <p><strong>Характеристики:</strong></p>
                <ul>
                    <li><strong>1 продукт банка</strong></li>
                    <li><strong>Неактивен</strong></li>
                    <li><strong>Возраст 18-30 лет</strong></li>
                    <li><strong>Средний баланс</strong></li>
                </ul>
                <p><strong>Вероятность оттока: 10-20%</strong></p>
                <p><strong>Потенциал:</strong> Высокий потенциал удержания через активацию</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="conclusion-card conclusion-low-risk">
            <h4>Клиент низкого риска</h4>
            <div>
                <p><strong>Характеристики:</strong></p>
                <ul>
                    <li><strong>3+ продукта банка</strong></li>
                    <li><strong>Активен</strong> (IsActiveMember=1)</li>
                    <li><strong>Возраст 20-35 лет</strong></li>
                    <li><strong>Женский пол</strong></li>
                    <li><strong>Средний баланс</strong></li>
                </ul>
                <p><strong>Вероятность оттока: 1-5%</strong></p>
                <p><strong>Бизнес-значение:</strong> Лояльная база, потенциал для кросс-продаж</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="conclusion-card conclusion-business" style="margin-top: 15px">
            <h4>VIP клиент</h4>
            <div>
                <p><strong>Характеристики:</strong></p>
                <ul>
                    <li><strong>4+ продукта банка</strong></li>
                    <li><strong>Активен</strong></li>
                    <li><strong>Возраст 40-55 лет</strong></li>
                    <li><strong>Очень высокий баланс</strong></li>
                </ul>
                <p><strong>Вероятность оттока: 1-2%</strong></p>
                <p><strong>Ценность:</strong> Ключевой клиент, требует премиального обслуживания</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif section == "Бизнес-рекомендации":
    st.markdown('<h2 class="eda-subtitle">Бизнес-рекомендации и действия</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Приоритетные действия по группам</h3>
            <h4 style="color: #ff4d4f;">Высокий приоритет</h4>   
            <p><strong>Клиенты с 1 продуктом + неактивны</strong></p> 
            <ul>
                <li>Персональный звонок менеджера</li>
                <li>Предложение второго продукта со скидкой 20%</li>
                <li>Бонусная программа, увеличенный процент по вкладу</li>
                <li>Срочная активационная кампания</li>
            </ul>
            <h4 style="color: #ffc107;">Средний приоритет</h4>
            <p><strong>Клиенты 45+ лет</strong></p>
            <ul>
                <li>Специальные условия для старшего возраста</li>
                <li>Персональный финансовый советник</li>
                <li>Консультации по пенсионным накоплениям</li>
                <li>Программа лояльности для постоянных клиентов</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Стратегические рекомендации</h3>
            <div>
                <h4 style="color: #2ebd85;">Низкий приоритет</h4>
                <p><strong>Клиенты с 2+ продуктами, активны</strong></p>
                <ul>
                    <li>Стандартные программы лояльности</li>
                    <li>Рекомендации премиум-услуг</li>
                    <li>Улучшение цифрового опыта</li>
                    <li>Программа рекомендаций друзьям</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #5294e2;">Проактивные меры</h4>
                <p><strong>Для всех клиентов</strong></p>
                <ul>
                    <li>Регулярный мониторинг активности</li>
                    <li>Персонализированные предложения</li>
                    <li>Уведомления о новых продуктах</li>
                    <li>Анализ удовлетворенности</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card">
        <h4>Матрица принятия решений</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
            <div>
                <strong>По количеству продуктов</strong>
                <p>1 продукт → Кросс-продажи<br>2 продукта → Удержание<br>3+ продукта → Лояльность</p>
            </div>
            <div>
                <strong>По активности</strong>
                <p>Неактивен → Реактивация<br>Активен → Поощрение<br>Очень активен → Премиум</p>
            </div>
            <div>
                <strong>По демографии</strong>
                <p>Молодой → Цифровизация<br>Средний возраст → Семейные пакеты<br>Старший → Надежность</p>
            </div>
            <div>
                <strong>По балансу</strong>
                <p>Низкий → Мотивация пополнений<br>Средний → Инвестиции<br>Высокий → Private banking</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Футер
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="text-align: center; color: #6c727d; margin: 0;">
        Интерпретация выполнена с использованием SHAP анализа и бизнес-аналитики
    </p>
</div>
""", unsafe_allow_html=True)
