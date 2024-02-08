import streamlit as st
import subprocess
import pandas as pd
from datetime import datetime
import os

# Assuming these scripts are in the same directory as the Streamlit app
classifier_script_path = './classifier.py'
recognizer_script_path = './recognizer.py'
attendance_file_path = './attendance.xlsx'

# Function to run a Python script with an argument and capture its output
def run_script_with_arg(script_path, arg):
    result = subprocess.run(['python', script_path, arg], capture_output=True, text=True)
    return result.stdout + '\n' + result.stderr


# Function to run a Python script and capture its output
def run_script(script_path):
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    return result.stdout + '\n' + result.stderr



# Initialize session state for button click
if 'load_clicked' not in st.session_state:
    st.session_state.load_clicked = False

# Streamlit UI
st.title('Facial Attendance System')

# Button to trigger name input
if st.button('Load Your Face'):
    st.session_state.load_clicked = True

# Ask for the name only if the button has been clicked
if st.session_state.load_clicked:
    name = st.text_input("Enter the name of the person:", "")

    if name:  # Run script only if name is provided
        output = run_script_with_arg(classifier_script_path, name)
        st.text_area('Output:', value=output, height=300)
        st.session_state.load_clicked = False  # Reset button state
    else:
        st.warning("Please enter a name to proceed.")


if st.button('Mark Attendance'):
    # Running the recognizer script
    output = run_script(recognizer_script_path)
    st.text_area('Output:', value=output, height=300)
    st.success(f'Attendance marked successfully.')

