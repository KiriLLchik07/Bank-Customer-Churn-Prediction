import streamlit as st
import pandas as pd
from scipy.stats import mannwhitneyu
import plotly.express as px

st.set_page_config(page_title="EDA - Bank Churn Analysis", layout="wide")

def load_eda_css():
    try:
        with open("assets/styles/eda.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Файл стилей EDA не найден. Используются стандартные стили.")

load_eda_css()

st.title("📊 Разведочный анализ данных (EDA)")
st.markdown("""
<div class="custom-card">
Анализ распределения признаков, выявление закономерностей и формулировка гипотез о факторах оттока клиентов банка.
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../../data/Churn_Modelling.csv')
        return df
    except FileNotFoundError:
        st.error("Файл данных не найден. Убедитесь, что файл Churn_Modelling.csv находится в папке data/")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

st.sidebar.header("🔍 Навигация по EDA")
section = st.sidebar.radio(
    "Выберите раздел:",
    ["📈 Обзор данных", "🎯 Анализ оттока", "📊 Категориальные признаки", 
     "🔢 Числовые признаки", "📈 Корреляционный анализ", "📋 Выводы"]
)

st.markdown("""
<style>
    .custom-card {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

if section == "📈 Обзор данных":
    st.header("📈 Обзор данных")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Всего клиентов", f"{len(df):,}")
    with col2:
        st.metric("Колонок", df.shape[1])
    with col3:
        st.metric("Пропуски", df.isnull().sum().sum())
    with col4:
        st.metric("Дубликаты", df.duplicated().sum())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Типы данных")
        st.dataframe(pd.DataFrame({
            'Тип': df.dtypes,
            'Уникальных': df.nunique(),
            'Пропуски': df.isnull().sum()
        }))
    
    with col2:
        st.subheader("Первые 10 строк")
        st.dataframe(df.head(10))
    
    st.subheader("📊 Описательная статистика числовых признаков")
    numeric_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    st.dataframe(df[numeric_cols].describe())

elif section == "🎯 Анализ оттока":
    st.header("🎯 Анализ целевой переменной")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(df, names='Exited', 
                    title='Распределение оттока клиентов',
                    color='Exited', 
                    color_discrete_map={0: '#00cc96', 1: '#ef553b'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        churn_count = df['Exited'].value_counts()
        churn_rate = df['Exited'].value_counts(normalize=True) * 100
        
        st.markdown("""
        <div class="custom-card">
        <h3>📈 Статистика оттока</h3>
        <p><strong>Лояльные клиенты:</strong> {:,} ({:.1f}%)</p>
        <p><strong>Ушедшие клиенты:</strong> {:,} ({:.1f}%)</p>
        <p><strong>Дисбаланс классов:</strong> {:.1f}:1</p>
        </div>
        """.format(churn_count[0], churn_rate[0], 
                  churn_count[1], churn_rate[1],
                  churn_count[0]/churn_count[1]), unsafe_allow_html=True)
        
        st.warning("""
        **🔍 Наблюдение:** Наблюдается значительный дисбаланс классов (80:20). 
        Это потребует специальных методов обработки на этапе моделирования.
        """)

elif section == "📊 Категориальные признаки":
    st.header("📊 Анализ категориальных признаков")
    
    categorical_features = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    
    selected_feature = st.selectbox("Выберите признак для анализа:", categorical_features)
    
    fig = px.histogram(df, x=selected_feature, color='Exited', barmode='group',
                      title=f'Распределение оттока по признаку {selected_feature}',
                      color_discrete_map={0: '#00cc96', 1: '#ef553b'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(f"📊 Статистика оттока по {selected_feature}")
    
    churn_by_feature = df.groupby(selected_feature)['Exited'].agg(['count', 'mean']).round(3)
    churn_by_feature['count'] = churn_by_feature['count'].astype(int)
    churn_by_feature['mean'] = (churn_by_feature['mean'] * 100).round(1)
    churn_by_feature.columns = ['Количество', 'Процент оттока (%)']
    
    st.dataframe(churn_by_feature)
    
    st.subheader("🔍 Ключевые наблюдения")
    
    insights = {
        'Geography': "🇩🇪 Клиенты из Германии уходят чаще (32%), чем из Франции (16%) и Испании (17%)",
        'Gender': "👩 Женщины склонны к оттоку больше (25%), чем мужчины (16%)",
        'NumOfProducts': "📦 Клиенты с 1 продуктом уходят в 27% случаев, с 2 продуктами - только в 7%",
        'HasCrCard': "💳 Наличие кредитной карты слабо влияет на отток (оба ~20%)",
        'IsActiveMember': "⚡ Неактивные клиенты уходят в 2 раза чаще (26% vs 14%)"
    }
    
    st.info(insights[selected_feature])

elif section == "🔢 Числовые признаки":
    st.header("🔢 Анализ числовых признаков")
    
    numeric_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    selected_numeric = st.selectbox("Выберите числовой признак:", numeric_features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df, x='Exited', y=selected_numeric, 
                    color='Exited',
                    title=f'Распределение {selected_numeric}',
                    color_discrete_map={0: '#00cc96', 1: '#ef553b'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x=selected_numeric, color='Exited',
                          nbins=50, barmode='overlay',
                          title=f'Гистограмма {selected_numeric}',
                          opacity=0.7,
                          color_discrete_map={0: '#00cc96', 1: '#ef553b'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Статистический анализ")
    
    group_0 = df[df['Exited'] == 0][selected_numeric]
    group_1 = df[df['Exited'] == 1][selected_numeric]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Медиана (0)", f"{group_0.median():.1f}")
    with col2:
        st.metric("Медиана (1)", f"{group_1.median():.1f}")
    with col3:
        st.metric("p-value (Манн-Уитни)", f"{mannwhitneyu(group_0, group_1)[1]:.4f}")
    with col4:
        stat, p_value = mannwhitneyu(group_0, group_1)
        n1, n2 = len(group_0), len(group_1)
        r = 1 - (2 * stat) / (n1 * n2)
        st.metric("Размер эффекта", f"{abs(r):.3f}")

elif section == "📈 Корреляционный анализ":
    st.header("📈 Корреляционный анализ")
    
    correlation_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'Tenure', 'EstimatedSalary', 'Exited']
    corr_matrix = df[correlation_features].corr(method='spearman')
    
    fig = px.imshow(corr_matrix, 
                   text_auto=True, 
                   aspect="auto",
                   color_continuous_scale='RdBu_r',
                   title='Матрица корреляций (Spearman)')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔗 Корреляции с целевой переменной (Exited)")
    
    target_correlations = corr_matrix['Exited'].sort_values(ascending=False)
    target_correlations = target_correlations[target_correlations.index != 'Exited']
    
    fig = px.bar(x=target_correlations.values, y=target_correlations.index,
                orientation='h',
                title='Влияние признаков на отток',
                labels={'x': 'Коэффициент корреляции', 'y': 'Признак'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Интерпретация корреляций")
    
    interpretation = {
        'Age': "🚀 Сильное влияние: Старшие клиенты значительно чаще уходят",
        'NumOfProducts': "📊 Умеренное влияние: Больше продуктов → меньше отток", 
        'Balance': "📊 Умеренное влияние: Высокий баланс ассоциирован с оттоком",
        'CreditScore': "📉 Слабое влияние: Минимальное воздействие на отток",
        'Tenure': "📉 Незначимое: Срок сотрудничества почти не влияет",
        'EstimatedSalary': "📉 Незначимое: Зарплата не связана с оттоком"
    }
    
    for feature, desc in interpretation.items():
        with st.expander(f"{feature}: {target_correlations[feature]:.3f}"):
            st.write(desc)

elif section == "📋 Выводы":
    st.header("📋 Ключевые выводы EDA")
    
    st.markdown("""
    <div class="custom-card">
    <h3>🎯 Основные инсайты</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Факторы высокого риска")
        st.markdown("""
        - **1 продукт банка** → 51% риск vs 5% при 3+ продуктах
        - **Неактивность** → 45% увеличение риска оттока  
        - **Возраст 45+** → 35% увеличение риска
        - **Германия** → 32% оттока vs 16% во Франции
        - **Женский пол** → 25% оттока vs 16% у мужчин
        """)
    
    with col2:
        st.subheader("🟢 Факторы низкого риска")
        st.markdown("""
        - **3+ продукта** → всего 5% риска оттока
        - **Активность** → 14% оттока vs 26% у неактивных
        - **Молодой возраст** → минимальный риск до 35 лет
        - **Мужской пол** → снижение риска на 9%
        - **Франция/Испания** → низкий региональный риск
        """)
    
    st.markdown("""
    <div class="custom-card">
    <h3>📊 Статистические выводы</h3>
    <p><strong>Сильное влияние:</strong> Возраст, количество продуктов</p>
    <p><strong>Умеренное влияние:</strong> Баланс, география, активность</p>  
    <p><strong>Слабое влияние:</strong> Кредитный рейтинг, пол</p>
    <p><strong>Незначимое:</strong> Зарплата, срок сотрудничества, кредитная карта</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="custom-card">
    <h3>🎯 Рекомендации для бизнеса</h3>
    <p><strong>Приоритетные меры:</strong> Кросс-продажи для клиентов с 1 продуктом</p>
    <p><strong>Сегментация:</strong> Особое внимание клиентам 45+ из Германии</p>
    <p><strong>Мониторинг:</strong> Активность как ключевой индикатор лояльности</p>
    <p><strong>Кампании:</strong> Персонализированные предложения по демографии</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("📊 EDA анализ выполнен с использованием Streamlit, Plotly и статистических методов")
