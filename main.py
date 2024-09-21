"""Main Driver"""

import streamlit as st

st.write(
    """
# Welcome to the Family Faces app!"""
)

in_zipfile_mode = st.toggle(label="Upload a Zip folder with photos instead of a set of photos.")

if in_zipfile_mode:
    images = st.file_uploader(
        label="Please upload a zip folder containing your images",
        type=["ZIP"],
    )
else:
    images = st.file_uploader(
        label="Please upload a zip folder containing your images",
        type=["PNG", "JPEG", "JPG"],
    )

ai_button_pressed = st.button(label="Start Image Recognition")
