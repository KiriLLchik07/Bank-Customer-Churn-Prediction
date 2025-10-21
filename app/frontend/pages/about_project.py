import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render():
    st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <h1>🏦 Прогнозирование оттока клиентов банка</h1>
        <p style='font-size: 1.2rem; color: #666;'>
            ML система для идентификации клиентов с высоким риском ухода и персонализированных рекомендаций по удержанию
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Быстрая статистика проекта
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Точность модели", 
            value="87.2%",
            delta="ROC-AUC"
        )
    
    with col2:
        st.metric(
            label="📊 Размер датасета", 
            value="10,000+",
            delta="клиентов"
        )
    
    with col3:
        st.metric(
            label="🤖 Лучшая модель", 
            value="CatBoost",
            delta="с тюнингом"
        )
    
    with col4:
        st.metric(
            label="⚡ Быстрота предсказания", 
            value="< 1 сек",
            delta="на клиента"
        )
    
    st.markdown("---")
    
    # Основные разделы
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Ценность для бизнеса")
        
        business_metrics = {
            "📉 Снижение оттока": "20-30% через превентивные меры",
            "🎯 Персонализация": "Кампании удержания для разных сегментов", 
            "💰 Экономия бюджета": "Снижение затрат на привлечение новых клиентов",
            "📈 Увеличение LTV": "Рост пожизненной ценности клиента"
        }
        
        for metric, value in business_metrics.items():
            with st.expander(f"{metric}"):
                st.write(value)
    
    with col2:
        st.subheader("🔍 Ключевые факторы оттока")
        
        factors = [
            {"factor": "1 продукт банка", "risk": "51% риск vs 5% при 3+ продуктах"},
            {"factor": "Неактивность клиента", "risk": "45% увеличение риска"},
            {"factor": "Возраст 45+ лет", "risk": "35% увеличение риска"}, 
            {"factor": "Гендер (мужчины)", "risk": "Выше склонность к оттоку"},
            {"factor": "Высокий баланс", "risk": "Риск потери VIP-клиентов"}
        ]
        
        for item in factors:
            st.write(f"• **{item['factor']}** → {item['risk']}")
    
    st.markdown("---")
    
    # Архитектура проекта
    st.subheader("🏗️ Архитектура проекта")
    
    tab1, tab2, tab3 = st.tabs(["📁 Структура", "🛠️ Технологии", "🚀 Roadmap"])
    
    with tab1:
        st.markdown("""
        ```
        Bank-Customer-Churn-Prediction/
        ├── ⚙️ config/               # Конфигурационные файлы
        ├── 📊 data/                 # Исходные и обработанные данные  
        ├── 🤖 models/               # Обученные модели
        ├── 📓 notebooks/            # Jupyter notebooks для анализа
        ├── 📈 reports/              # Отчеты и визуализации
        ├── 🔧 src/                  # Исходный код
        └── 🐳 app/                  # FastAPI и Streamlit приложения
        ```
        """)
    
    with tab2:
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.write("**🤖 Машинное обучение:**")
            st.write("- CatBoost, LightGBM, XGBoost")
            st.write("- Scikit-learn, Imbalanced-learn") 
            st.write("- Optuna для тюнинга")
            st.write("- SHAP для интерпретации")
            
        with tech_col2:
            st.write("**🌐 Веб-разработка:**")
            st.write("- FastAPI для бэкенда")
            st.write("- Streamlit для фронтенда")
            st.write("- Poetry для управления зависимостями")
            st.write("- Docker для контейнеризации")
    
    with tab3:
        roadmap_data = {
            "✅ Завершено": [
                "EDA и анализ данных",
                "Feature Engineering", 
                "Обучение и сравнение моделей",
                "Интерпретация моделей (SHAP)"
            ],
            "🔄 В процессе": [
                "FastAPI для продакшн использования",
                "Streamlit интерфейс для бизнес-пользователей"
            ],
            "⏳ Планируется": [
                "Docker контейнеризация",
                "Интеграция с реальными данными",
                "A/B тестирование рекомендаций"
            ]
        }
        
        for status, items in roadmap_data.items():
            with st.expander(f"{status} ({len(items)} задач)"):
                for item in items:
                    st.write(f"• {item}")
    
    st.markdown("---")
    
    # Призыв к действию
    st.success("""
    **🚀 Начните использовать систему прямо сейчас!** 
    
    Перейдите во вкладку **🎯 Предсказание оттока** чтобы:
    - Проанализировать конкретного клиента
    - Получить оценку риска оттока  
    - Увидеть персонализированные рекомендации по удержанию
    """)

if __name__ == "__main__":
    render()
