import streamlit as st
import sys
sys.path.append('../../')

from back_end.profile import *
def write():
    st.title("Login Page: Please Login into No More COVID-Confusion.")
    username = st.text_input("Enter your Username: ")
    password = st.text_input("Enter your Password: ", type = 'password')
    login = st.button("Login!")
    register = st.button("Register!")
    if login:
        verified = DataBase.verify(username, password)
        profile = DataBase.retrieve(username)
        if verified:
            return profile
        st.write("Incorrect Username or Password.")
    elif register:
        exists = DataBase.exists(username)
        if not exists:
            DataBase.register(username, password)
            profile = DataBase.retrieve(username)
            return profile
        else:
            st.write("Username already registered. Please Pick another.")
    return None