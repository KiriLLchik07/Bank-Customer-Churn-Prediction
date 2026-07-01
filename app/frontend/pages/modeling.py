import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Моделирование - Bank Churn", 
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
    <h1 class="eda-title">Моделирование и эксперименты</h1>
    <div class="eda-card">
        <p style="text-align: center; font-size: 1.2rem; margin: 0;">
            Сравнение ML-алгоритмов, оптимизация гиперпараметров и выбор лучшей модели для прогнозирования оттока клиентов.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-nav">
    <h3 style="margin-bottom: 1rem; border-bottom: 1px solid #2a2f38; padding-bottom: 0.5rem;">Навигация по моделированию</h3>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "Выберите раздел:",
    ["Обзор эксперимента", "Сравнение моделей", "Оптимизация гиперпараметров", 
     "Анализ результатов", "Важность признаков", "Финальные выводы"],
    label_visibility="collapsed"
)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

@st.cache_data
def load_modeling_data():
    model_comparison = {
        'Модель': ['CatBoost (тюнинг)', 'LightGBM (тюнинг)', 'CatBoost (базовый)', 
                  'Random Forest', 'LightGBM (базовый)', 'XGBoost', 'Logistic Regression'],
        'ROC-AUC': [0.8720, 0.8633, 0.8628, 0.8468, 0.8455, 0.8389, 0.7592],
        'F1-Score': [0.6350, 0.5893, 0.5901, 0.5807, 0.5722, 0.5655, 0.5218],
        'Precision': [0.6166, 0.6024, 0.5943, 0.5938, 0.5876, 0.5812, 0.5341],
        'Recall': [0.6544, 0.5768, 0.5859, 0.5681, 0.5574, 0.5512, 0.5102],
        'Переобучение (AUC diff)': [0.0183, 0.0296, 0.1003, 0.1532, 0.1412, 0.1289, 0.0456]
    }
    
    feature_importance = {
        'Признак': ['NumOfProducts', 'Age', 'Balance', 'IsActiveMember', 'Geography_Germany', 
                   'Gender_Male', 'CreditScore', 'Geography_Spain', 'Tenure', 'EstimatedSalary'],
        'Важность': [24.3, 18.7, 15.2, 12.8, 8.5, 6.3, 5.1, 4.2, 3.1, 1.8]
    }
    
    catboost_params = {
        'Параметр': ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg', 
                    'border_count', 'random_strength', 'bagging_temperature'],
        'Значение': [274, 4, 0.0215, 2.293, 215, 0.317, 0.489],
        'Описание': [
            'Количество деревьев в ансамбле',
            'Максимальная глубина деревьев',
            'Скорость обучения',
            'L2 регуляризация',
            'Количество разбиений для числовых признаков',
            'Случайность при выборе разбиений',
            'Интенсивность бутстраппинга'
        ]
    }
    
    return pd.DataFrame(model_comparison), pd.DataFrame(feature_importance), pd.DataFrame(catboost_params)

model_comparison_df, feature_importance_df, catboost_params_df = load_modeling_data()

if section == "Обзор эксперимента":
    st.markdown('<h2 class="eda-subtitle">Обзор эксперимента</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Цели этапа</h3>
            <ul>
                <li>Обучить и сравнить различные ML-алгоритмы</li>
                <li>Провести подбор гиперпараметров для лучших моделей</li>
                <li>Выбрать оптимальную модель для прогнозирования оттока</li>
                <li>Проанализировать переобучение и стабильность моделей</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Метрики оценки</h3>
            <ul>
                <li><strong>ROC-AUC</strong> - Качество разделения классов (целевая метрика)</li>
                <li><strong>F1-Score</strong> - Баланс между точностью и полнотой</li>
                <li><strong>Precision</strong> - Точность положительных предсказаний</li>
                <li><strong>Recall</strong> - Полнота обнаружения уходящих клиентов</li>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Использованные алгоритмы</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <h4 style="color: #a0a5b0;">Базовые модели</h4>
                    <ul>
                        <li>Logistic Regression</li>
                        <li>K-Neighbors</li>
                        <li>Decision Tree</li>
                        <li>Random Forest</li>
                    </ul>
                </div>
                <div>
                    <h4 style="color: #a0a5b0;">Продвинутые ансамбли</h4>
                    <ul>
                        <li>XGBoost</li>
                        <li>LightGBM</li>
                        <li>CatBoost</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Процесс экспериментов</h3>
            <ol>
                <li>Базовое сравнение 7 алгоритмов</li>
                <li>Гиперпараметрический поиск с Optuna</li>
                <li>Анализ переобучения на train/test</li>
                <li>Оптимизация порога классификации</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

elif section == "Сравнение моделей":
    st.markdown('<h2 class="eda-subtitle">Сравнение моделей</h2>', unsafe_allow_html=True)
    
    st.subheader("Сравнительная таблица моделей")
    st.dataframe(model_comparison_df.style.format({
        'ROC-AUC': '{:.4f}',
        'F1-Score': '{:.4f}', 
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'Переобучение (AUC diff)': '{:.4f}'
    }).highlight_max(subset=['ROC-AUC', 'F1-Score'], color='#2ebd85')
                 .highlight_min(subset=['Переобучение (AUC diff)'], color='#2ebd85'), 
                 use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plotly-chart-container"><h4>Сравнение ROC-AUC моделей</h4>', unsafe_allow_html=True)
        fig = px.bar(model_comparison_df, x='Модель', y='ROC-AUC',
                    color='ROC-AUC',
                    color_continuous_scale='Viridis')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a5b0',
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plotly-chart-container"><h4>Анализ переобучения моделей</h4>', unsafe_allow_html=True)
        fig = px.bar(model_comparison_df, x='Модель', y='Переобучение (AUC diff)',
                    color='Переобучение (AUC diff)',
                    color_continuous_scale='RdBu_r')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a5b0',
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card insight-strong">
        <h4>Ключевые наблюдения</h4>
        <ul>
            <li><strong>CatBoost показал наилучшие результаты</strong> после тюнинга гиперпараметров</li>
            <li><strong>Ансамблевые методы</strong> превосходят базовые алгоритмы</li>
            <li><strong>Переобучение минимально</strong> у тюнингованных моделей</li>
            <li><strong>LightGBM</strong> демонстрирует хороший баланс качества и скорости</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif section == "Оптимизация гиперпараметров":
    st.markdown('<h2 class="eda-subtitle">Оптимизация гиперпараметров</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Метод оптимизации</h3>
            <div>
                <p><strong>Библиотека:</strong> Optuna</p>
                <p><strong>Алгоритм:</strong> TPE (Tree-structured Parzen Estimator)</p>
                <p><strong>Количество trials:</strong> 100 на модель</p>
                <p><strong>Метрика:</strong> ROC-AUC (кросс-валидация)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Преимущества Optuna</h3>
            <ul>
                <li><strong>Быстрее</strong> чем GridSearch</li>
                <li><strong>Точнее</strong> чем RandomSearch</li>
                <li><strong>Интеллектуальный</strong> поиск параметров</li>
                <li><strong>Визуализация</strong> процесса оптимизации</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="eda-card"><h4>Оптимальные параметры CatBoost</h4>', unsafe_allow_html=True)
        st.dataframe(catboost_params_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="eda-card">
        <h3 style="color: #b8860b;"> Процесс оптимизации гиперпараметров</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; color: #a0a5b0;">
            <div style="text-align: center;">
                <div style="font-size: 2rem;">1</div>
                <strong>Инициализация</strong>
                <p>Определение пространства параметров</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">2</div>
                <strong>Сэмплирование</strong>
                <p>TPE выбирает перспективные параметры</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">3</div>
                <strong>Оценка</strong>
                <p>Обучение и валидация модели</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">4</div>
                <strong>Обновление</strong>
                <p>Алгоритм учится на результатах</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif section == "Анализ результатов":
    st.markdown('<h2 class="eda-subtitle">Анализ результатов</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #2ebd85;">Финальные метрики CatBoost</h3>
            <div class="eda-metrics">
                <div class="eda-metric">
                    <div class="eda-metric-value">0.872</div>
                    <div class="eda-metric-label">ROC-AUC</div>
                </div>
                <div class="eda-metric">
                    <div class="eda-metric-value">0.635</div>
                    <div class="eda-metric-label">F1-Score</div>
                </div>
                <div class="eda-metric">
                    <div class="eda-metric-value">0.617</div>
                    <div class="eda-metric-label">Precision</div>
                </div>
                <div class="eda-metric">
                    <div class="eda-metric-value">0.654</div>
                    <div class="eda-metric-label">Recall</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Оптимизация порога</h3>
            <div>
                <p><strong>Оптимальный порог:</strong> 0.640 (вместо 0.5 по умолчанию)</p>
                <p><strong>Метод:</strong> Максимизация F1-Score</p>
                <p><strong>Бизнес-интерпретация:</strong></p>
                <ul>
                    <li><strong>Высокий Recall</strong> (65%) - находим большинство уходящих</li>
                    <li><strong>Умеренный Precision</strong> (62%) - минимизируем ложные тревоги</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Матрица ошибок</h3>
            <div style="text-align: center; color: #a0a5b0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin: 1rem 0;">
                    <div style="background: rgba(46, 189, 133, 0.2); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #2ebd85;">1427</div>
                        <div>True Negative</div>
                    </div>
                    <div style="background: rgba(255, 77, 79, 0.2); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #ff4d4f;">166</div>
                        <div>False Positive</div>
                    </div>
                    <div style="background: rgba(255, 77, 79, 0.2); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #ff4d4f;">141</div>
                        <div>False Negative</div>
                    </div>
                    <div style="background: rgba(46, 189, 133, 0.2); padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #2ebd85;">267</div>
                        <div>True Positive</div>
                    </div>
                </div>
                <p><strong>Точность:</strong> 84.7% | <strong>Полнота:</strong> 65.4%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif section == "Важность признаков":
    st.markdown('<h2 class="eda-subtitle">Важность признаков</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="plotly-chart-container"><h4>Топ-10 самых важных признаков</h4>', unsafe_allow_html=True)
        fig = px.bar(feature_importance_df, x='Важность', y='Признак',
                    orientation='h',
                    color='Важность',
                    color_continuous_scale='Viridis')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a5b0',
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="eda-card">
            <h3 style="color: #b8860b;">Ключевые факторы</h3>
            <div style="color: #a0a5b0;">
                <div style="margin-bottom: 1rem;">
                    <strong>Топ-5 признаков:</strong>
                    <ol>
                        <li>NumOfProducts (24.3%)</li>
                        <li>Age (18.7%)</li>
                        <li>Balance (15.2%)</li>
                        <li>IsActiveMember (12.8%)</li>
                        <li>Geography_Germany (8.5%)</li>
                    </ol>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card">
        <h4>Бизнес-интерпретация важности признаков</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
            <div>
                <strong>Количество продуктов</strong>
                <p>Клиенты с 1 продуктом в 5 раз чаще уходят</p>
            </div>
            <div>
                <strong>Возраст</strong>
                <p>Клиенты 45+ значительно чаще покидают банк</p>
            </div>
            <div>
                <strong>Баланс счета</strong>
                <p>Высокие балансы ассоциированы с риском оттока</p>
            </div>
            <div>
                <strong>Активность</strong>
                <p>Неактивные клиенты уходят в 2 раза чаще</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif section == "Финальные выводы":
    st.markdown('<h2 class="eda-subtitle">Финальные выводы</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="conclusion-card conclusion-high-risk" style="margin-top:15px;">
            <h4>Достигнутые цели</h4>
            <ul>
                <li>Выбрана оптимальная модель - CatBoost</li>
                <li>Оптимизированы гиперпараметры с Optuna</li>
                <li>Достигнуты отличные метрики качества</li>
                <li>Проанализирована важность признаков</li>
                <li>Найден оптимальный порог классификации</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="conclusion-card conclusion-business" style="margin-top:15px">
            <h4>Практическая ценность</h4>
            <ul>
                <li><strong>ROC-AUC 0.872</strong> - отличное качество предсказаний</li>
                <li><strong>F1-Score 0.635</strong> - хороший баланс метрик</li>
                <li><strong>65% Recall</strong> - находим большинство уходящих</li>
                <li><strong>Экономический эффект</strong> - снижение оттока на 20-30%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="conclusion-card conclusion-statistical" style="margin-top:15px">
            <h4>Рекомендации по использованию</h4>
            <div>
                <p><strong>Продакшн-развертывание:</strong></p>
                <ul>
                    <li>Использовать CatBoost с оптимизированными параметрами</li>
                    <li>Применять порог классификации 0.64</li>
                    <li>Мониторить качество модели на новых данных</li>
                    <li>Регулярно переобучать на актуальных данных</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="conclusion-card conclusion-statistical" style="margin-top:15px">                   
                <h4>Бизнес-применение</h4>
                <ul>
                    <li>Фокус на клиентах с 1 продуктом</li>
                    <li>Особое внимание клиентам 45+ из Германии</li>
                    <li>Мониторинг активности клиентов</li>
                    <li>Персонализированные кампании удержания</li>
                </ul>
""", unsafe_allow_html=True)
    
    # Финальная статистика
    st.markdown("""
    <div class="eda-card" style="text-align: center;">
        <h3>Итоги моделирования</h3>
        <div class="eda-metrics">
            <div class="eda-metric">
                <div class="eda-metric-value">7</div>
                <div class="eda-metric-label">Протестировано моделей</div>
            </div>
            <div class="eda-metric">
                <div class="eda-metric-value">200+</div>
                <div class="eda-metric-label">Экспериментов с параметрами</div>
            </div>
            <div class="eda-metric">
                <div class="eda-metric-value">0.872</div>
                <div class="eda-metric-label">Лучший ROC-AUC</div>
            </div>
            <div class="eda-metric">
                <div class="eda-metric-value">CatBoost</div>
                <div class="eda-metric-label">Победитель</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Футер
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="text-align: center; color: #6c727d; margin: 0;">
        Моделирование выполнено с использованием CatBoost, LightGBM, XGBoost и Optuna
    </p>
</div>
""", unsafe_allow_html=True)
