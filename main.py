"""Main Driver"""

import streamlit as st

st.write(
    """
# Welcome to the Family Faces app!"""
)


images = st.file_uploader(
    label="Please upload a zip folder containing your images",
    type=["PNG", "JPEG", "JPG"],
    accept_multiple_files=True
)
