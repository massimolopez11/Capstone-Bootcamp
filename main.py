"""Main Driver"""

import os
import zipfile

import streamlit as st

import tools

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for images folder. If not create one
try:
    os.mkdir("images")
    logger.debug("New directory called `images`")
except FileExistsError as e:
    logger.debug("%s: `images`", e.strerror)

st.write(
    """
# Welcome to the Family Faces app!"""
)

in_zipfile_mode = st.toggle(
    label="Upload a Zip folder with photos instead of a set of photos."
)

if in_zipfile_mode:
    images = st.file_uploader(
        label="Please upload a zip folder containing your images",
        type=["ZIP"],
    )

    if images is not None:
        with zipfile.ZipFile(images, "r") as zip_ref:
            zip_ref.extractall("images/.")

    # Check if all files are images and remove invalid file types
    for dirpath, _, files in os.walk("images/"):
        for filename in files:
            if not tools.is_valid_image(f"{dirpath}/{filename}"):
                os.remove(f"{dirpath}/{filename}")
else:
    images = st.file_uploader(
        label="Please upload a zip folder containing your images",
        type=["PNG", "JPEG", "JPG"],
        accept_multiple_files=True
    )

    if images is not None:
        for image in images:
            with open(os.path.join("images/", image.name), "wb") as f:
                f.write(image.getbuffer())


ai_button_pressed = st.button(label="Start Image Recognition")

if ai_button_pressed:
    # TODO: fit the model here.
    pass
