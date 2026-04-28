import streamlit as st
# Весь наш код ассистента сюда
user_input = st.text_input("Задайте вопрос по страхованию:")
if user_input:
    st.write(ask(user_input))
