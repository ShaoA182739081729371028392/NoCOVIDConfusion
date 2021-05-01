# Handles Logic After User has Logged In.
import streamlit as st
import sys
sys.path.append('../../')
import datetime
def write(profile):
    cur_time = datetime.datetime.now()
    date_string = cur_time.strftime('%H: %M')
    # Extract Data from Profile
    username = profile['_id']
    visited = profile['visited']
    quarantine = profile['quarantine']
    st.title(f"Hello {username}!      Current Time: {date_string}")
    st.header("Welcome to the No-COVID-Confusion Portal.")
    st.write("Diagnosed with COVID-19? Hit the Quarantine Button to start your quarantine timer!")
    quarantine = st.button("Quarantine.")
    if quarantine:
        profile['quarantine'] = 60 * 60 * 24 * 14 # 2 Week Quarantine
    phrase = "Your Quarantine is Over. Stay Safe!" if round(profile['quarantine'] // 60**2) <= 0 else "Please stay inside. Your Quarantine still continues." 
    st.write(f"Quarantine Time Left(in hours): {round(profile['quarantine'] // 60**2)}. {phrase}")
    # Places I visited
    st.write("Places I Visited List. Keep track of where you visited to stay safe.")
    selected = st.multiselect("Places I Visited:", visited)
    delete = st.button("Delete Item")
    if delete:
        new_list = []
        for item in profile['visited']:
            if item not in selected:
                new_list += [item]
        profile['visited'] = new_list
    # Add an Item to the list
    text_input = st.text_input('Add a place: ')
    insert = st.button("Insert Item")
    if insert and text_input is not None and text_input != "":
        if text_input not in profile['visited']:
            profile['visited'] += [f'{text_input}, {date_string}']

    log_out = st.button("Logout.")
    if log_out:
        return 'logout'
    return profile