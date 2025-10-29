import streamlit as st

page_1 = st.Page("pages/about_project.py", title="Главная страница")
page_2 = st.Page("pages/EDA.py", title="Разведочный анализ данных")
page_3 = st.Page("pages/modeling.py", title="Моделирование и эксперименты")
page_4 = st.Page("pages/interpretation.py", title="Интерпретация модели")
page_5 = st.Page("pages/predictions.py", title="Предсказать отток")

pg = st.navigation([ page_1, page_2, page_3, page_4, page_5])

pg.run()
