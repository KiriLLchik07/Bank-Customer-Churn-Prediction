import streamlit as st
import pandas as pd
from scipy.stats import mannwhitneyu
import plotly.express as px

st.set_page_config(
    page_title="EDA - Bank Churn Analysis", 
    layout="wide"
)

def load_css():
    try:
        with open("assets/styles/eda.css", "r", encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Файл стилей EDA не найден")

load_css()

st.markdown("""
<div class="eda-container">
    <h1 class="eda-title">Разведочный анализ данных (EDA)</h1>
    <div class="eda-card">
        <p style="text-align: center; font-size: 1.2rem; margin: 0;">
            Анализ распределения признаков, выявление закономерностей и формулировка гипотез о факторах оттока клиентов банка.
        </p>
    </div>
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

st.sidebar.markdown("""
<div class="sidebar-nav">
    <h3 style="margin-bottom: 1rem; border-bottom: 1px solid #2a2f38; padding-bottom: 0.5rem;">Навигация по EDA</h3>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "Выберите раздел:",
    ["Обзор данных", "Анализ оттока", "Категориальные признаки", 
     "Числовые признаки", "Корреляционный анализ", "Выводы"],
    label_visibility="collapsed"
)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

if section == "Обзор данных":
    st.markdown('<h2 class="eda-subtitle">Обзор данных</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="eda-metrics">
        <div class="eda-metric">
            <div class="eda-metric-value">{:,}</div>
            <div class="eda-metric-label">Всего клиентов</div>
        </div>
        <div class="eda-metric">
            <div class="eda-metric-value">{}</div>
            <div class="eda-metric-label">Колонок</div>
        </div>
        <div class="eda-metric">
            <div class="eda-metric-value">{}</div>
            <div class="eda-metric-label">Пропуски</div>
        </div>
        <div class="eda-metric">
            <div class="eda-metric-value">{}</div>
            <div class="eda-metric-label">Дубликаты</div>
        </div>
    </div>
    """.format(len(df), df.shape[1], df.isnull().sum().sum(), df.duplicated().sum()), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="eda-card data-overview-card"><h5>Типы данных</h5></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Тип': df.dtypes,
            'Уникальных': df.nunique(),
            'Пропуски': df.isnull().sum()
        }), use_container_width=True)
    
    with col2:
        st.markdown('<div class="eda-card data-overview-card"><h5>Первые 10 строк</h5></div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown('<div class="eda-card data-overview-card"><h4>Описательная статистика числовых признаков</h4></div>', unsafe_allow_html=True)
    numeric_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

elif section == "Анализ оттока":
    st.markdown('<h2 class="eda-subtitle">Анализ целевой переменной</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plotly-chart-container"><h5>Распределение оттока клиентов</h5>', unsafe_allow_html=True)
        fig = px.pie(df, names='Exited', 
                    color='Exited', 
                    color_discrete_map={0: '#00cc96', 1: '#ef553b'})
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a5b0'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        churn_count = df['Exited'].value_counts()
        churn_rate = df['Exited'].value_counts(normalize=True) * 100
        
        st.markdown("""
        <div class="eda-card churn-analysis-card">
            <h3 style="margin-bottom: 1rem;">Статистика оттока</h3>
            <div class="eda-metrics">
                <div class="eda-metric">
                    <div class="eda-metric-value">{:,}</div>
                    <div class="eda-metric-label">Лояльные клиенты</div>
                </div>
                <div class="eda-metric">
                    <div class="eda-metric-value">{:,}</div>
                    <div class="eda-metric-label">Ушедшие клиенты</div>
                </div>
                <div class="eda-metric">
                    <div class="eda-metric-value">{:.1f}%</div>
                    <div class="eda-metric-label">Процент оттока</div>
                </div>
                <div class="eda-metric">
                    <div class="eda-metric-value">{:.1f}:1</div>
                    <div class="eda-metric-label">Дисбаланс</div>
                </div>
            </div>
        </div>
        """.format(churn_count[0], churn_count[1], churn_rate[1], churn_count[0]/churn_count[1]), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card insight-strong">
            <strong>Наблюдение:</strong> Наблюдается значительный дисбаланс классов (80:20). 
            Это потребует специальных методов обработки на этапе моделирования.
        </div>
        """, unsafe_allow_html=True)

elif section == "Категориальные признаки":
    st.markdown('<h2 class="eda-subtitle">Анализ категориальных признаков</h2>', unsafe_allow_html=True)
    
    categorical_features = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    
    selected_feature = st.selectbox("Выберите признак для анализа:", categorical_features)
    
    fig = px.histogram(df, x=selected_feature, color='Exited', barmode='group',
                      title=f'Распределение оттока по признаку {selected_feature}',
                      color_discrete_map={0: '#00cc96', 1: '#ef553b'})
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#a0a5b0'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(f"Статистика оттока по {selected_feature}")
    
    churn_by_feature = df.groupby(selected_feature)['Exited'].agg(['count', 'mean']).round(3)
    churn_by_feature['count'] = churn_by_feature['count'].astype(int)
    churn_by_feature['mean'] = (churn_by_feature['mean'] * 100).round(1)
    churn_by_feature.columns = ['Количество', 'Процент оттока (%)']
    
    st.dataframe(churn_by_feature, use_container_width=True)
    
    st.subheader("Ключевые наблюдения")
    
    insights = {
        'Geography': "Клиенты из Германии уходят чаще (32%), чем из Франции (16%) и Испании (17%)",
        'Gender': "Женщины склонны к оттоку больше (25%), чем мужчины (16%)",
        'NumOfProducts': "Клиенты с 1 продуктом уходят в 27% случаев, с 2 продуктами - только в 7%",
        'HasCrCard': "Наличие кредитной карты слабо влияет на отток (оба ~20%)",
        'IsActiveMember': "Неактивные клиенты уходят в 2 раза чаще (26% vs 14%)"
    }
    
    st.info(insights[selected_feature])

elif section == "Числовые признаки":
    st.markdown('<h2 class="eda-subtitle">Анализ числовых признаков</h2>', unsafe_allow_html=True)
    
    numeric_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    selected_numeric = st.selectbox("Выберите числовой признак:", numeric_features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df, x='Exited', y=selected_numeric, 
                    color='Exited',
                    title=f'Распределение {selected_numeric}',
                    color_discrete_map={0: '#00cc96', 1: '#ef553b'})
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a5b0'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x=selected_numeric, color='Exited',
                          nbins=50, barmode='overlay',
                          title=f'Гистограмма {selected_numeric}',
                          opacity=0.7,
                          color_discrete_map={0: '#00cc96', 1: '#ef553b'})
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a0a5b0'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="eda-card numerical-analysis-card"><h4>Статистический анализ</h4>', unsafe_allow_html=True)
    
    group_0 = df[df['Exited'] == 0][selected_numeric]
    group_1 = df[df['Exited'] == 1][selected_numeric]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Медиана (0)", f"{group_0.median():.1f}")
    with col2:
        st.metric("Медиана (1)", f"{group_1.median():.1f}")
    with col3:
        p_value = mannwhitneyu(group_0, group_1)[1]
        st.metric("p-value", f"{p_value:.4f}")
    with col4:
        stat, p_value = mannwhitneyu(group_0, group_1)
        n1, n2 = len(group_0), len(group_1)
        r = 1 - (2 * stat) / (n1 * n2)
        effect_strength = "Сильный эффект на целевую переменную" if abs(r) > 0.3 else "Умеренный эффект на целевую переменную" if abs(r) > 0.1 else "Слабый эффект на целевую переменную"
        st.metric("Размер эффекта", f"{abs(r):.3f}")
    
    st.markdown("""
    <div class="insight-card">
        <strong>Интерпретация:</strong> {}
    </div>
    """.format(effect_strength), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Корреляционный анализ":
    st.markdown('<h2 class="eda-subtitle">Корреляционный анализ</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="correlation-matrix"><h4>Матрица корреляций (Spearman)</h4>', unsafe_allow_html=True)
    correlation_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'Tenure', 'EstimatedSalary', 'Exited']
    corr_matrix = df[correlation_features].corr(method='spearman')
    
    fig = px.imshow(corr_matrix, 
                   text_auto=True, 
                   aspect="auto",
                   color_continuous_scale='RdBu_r')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#a0a5b0'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="eda-card correlation-analysis-card"><h4>Корреляции с целевой переменной (Exited)</h4>', unsafe_allow_html=True)
    
    target_correlations = corr_matrix['Exited'].sort_values(ascending=False)
    target_correlations = target_correlations[target_correlations.index != 'Exited']
    
    fig = px.bar(x=target_correlations.values, y=target_correlations.index,
                orientation='h',
                title='Влияние признаков на отток',
                labels={'x': 'Коэффициент корреляции', 'y': 'Признак'},
                color=target_correlations.values,
                color_continuous_scale='RdBu_r')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#a0a5b0',
        showlegend=False,
        yaxis={'categoryorder':'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-card"><h4>Интерпретация корреляций</h4>', unsafe_allow_html=True)
    
    interpretation = {
        'Age': "Сильное влияние: Старшие клиенты значительно чаще уходят",
        'NumOfProducts': "Умеренное влияние: Больше продуктов → меньше отток", 
        'Balance': "Умеренное влияние: Высокий баланс ассоциирован с оттоком",
        'CreditScore': "Слабое влияние: Минимальное воздействие на отток",
        'Tenure': "Незначимое: Срок сотрудничества почти не влияет",
        'EstimatedSalary': "Незначимое: Зарплата не связана с оттоком"
    }
    
    for feature, desc in interpretation.items():
        corr_value = target_correlations[feature]
        with st.expander(f"{feature}: {corr_value:.3f}"):
            st.write(desc)
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Выводы":
    st.markdown('<h2 class="eda-subtitle">Ключевые выводы EDA</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="eda-card conclusions-card">
        <h3 style="text-align: center; margin-bottom: 2rem;">Основные инсайты анализа</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="conclusion-card conclusion-high-risk">
            <h4>Факторы высокого риска</h4>
            <ul style=>
                <li><strong>1 продукт банка</strong> → 51% риск vs 5% при 3+ продуктах</li>
                <li><strong>Неактивность</strong> → 45% увеличение риска оттока</li>
                <li><strong>Возраст 45+</strong> → 35% увеличение риска</li>
                <li><strong>Германия</strong> → 32% оттока vs 16% во Франции</li>
                <li><strong>Женский пол</strong> → 25% оттока vs 16% у мужчин</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="conclusion-card conclusion-statistical" style="margin-top: 15px">
            <h4>Статистические выводы</h4>
            <ul style=>
            <li><strong>Сильное влияние:</strong> Возраст, количество продуктов</li>
            <li><strong>Умеренное влияние:</strong> Баланс, география, активность</li>  
            <li><strong>Слабое влияние:</strong> Кредитный рейтинг, пол</li>
            <li><strong>Незначимое:</strong> Зарплата, срок сотрудничества, кредитная карта</li>
        </div>
    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="conclusion-card conclusion-low-risk">
            <h4>Факторы низкого риска</h4>
            <ul style=>
                <li><strong>3+ продукта</strong> → всего 5% риска оттока</li>
                <li><strong>Активность</strong> → 14% оттока vs 26% у неактивных</li>
                <li><strong>Молодой возраст</strong> → минимальный риск до 35 лет</li>
                <li><strong>Мужской пол</strong> → снижение риска на 9%</li>
                <li><strong>Франция/Испания</strong> → низкий региональный риск</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="conclusion-card conclusion-business" style="margin-top: 15px">
            <h4>Рекомендации для бизнеса</h4>
            <ul style=>
            <li><strong>Приоритетные меры:</strong> Кросс-продажи для клиентов с 1 продуктом</li>
            <li><strong>Сегментация:</strong> Особое внимание клиентам 45+ из Германии</li>
            <li><strong>Мониторинг:</strong> Активность как ключевой индикатор лояльности</li>
            <li><strong>Кампании:</strong> Персонализированные предложения по демографии</li>
        </div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="text-align: center; color: #6c727d; margin: 0;">
        EDA анализ выполнен с использованием Streamlit, Plotly и статистических методов
    </p>
</div>
""", unsafe_allow_html=True)
