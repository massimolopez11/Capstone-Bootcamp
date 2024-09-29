"""Main Driver"""

import logging
import os
import shutil
import zipfile

import streamlit as st
from PIL import Image

import tools
from data_model import DataModels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for images folder to save images. If not create one.
try:
    os.mkdir("images")
    logger.debug("New directory called `images`")
except FileExistsError as e:
    logger.debug("%s: `images`", e.strerror)

# Empty the faces directory
CLUSTER_DIR = "clustered_faces"
if os.path.isdir(CLUSTER_DIR):
    shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR)
else:
    os.makedirs(CLUSTER_DIR)

IMAGES_DIR = "images"

# Empty the temp images directory
IMG_TEMP_DIR = "images_temp"
if os.path.isdir(IMG_TEMP_DIR):
    shutil.rmtree(IMG_TEMP_DIR)
    os.makedirs(IMG_TEMP_DIR)
else:
    os.makedirs(IMG_TEMP_DIR)


st.write(
    """
# Welcome to the Family Faces app!"""
)

in_zipfile_mode = st.toggle(
    label="Upload a Zip folder with photos instead of a set of photos."
)

if in_zipfile_mode:
    image_zip = st.file_uploader(
        label="Please upload a zip folder containing your images.",
        type=["ZIP"],
    )

    if image_zip is not None:
        with zipfile.ZipFile(image_zip, "r") as zip_ref:
            zip_ref.extractall("images/.")
            zip_ref.extractall("images_temp/.")

    # Check if all files are images and remove invalid file types
    for dirpath, _, files in os.walk("images/"):
        for filename in files:
            if not tools.is_valid_image(f"{dirpath}/{filename}"):
                os.remove(f"{dirpath}/{filename}")
    for dirpath, _, files in os.walk("images_temp/"):
        for filename in files:
            if not tools.is_valid_image(f"{dirpath}/{filename}"):
                os.remove(f"{dirpath}/{filename}")
else:
    images = st.file_uploader(
        label="Please upload a set of images.",
        type=["PNG", "JPEG", "JPG"],
        accept_multiple_files=True,
    )

    if images is not None:
        for image in images:
            with open(os.path.join("images/", image.name), "wb") as f:
                f.write(image.getbuffer())
    if images is not None:
        for image in images:
            with open(os.path.join("images_temp/", image.name), "wb") as f:
                f.write(image.getbuffer())


col1, col2, col3 = st.columns(3)
with col1:
    predict_button = st.button(label="Start Image Recognition")
with col2:
    train_button = st.button(
        label="Train Your Own Model",
        help="NOTE: This removes the pretrained model already generated.",
    )
# with st.stylable_container(
#     key="Upload_Data",
#     css_styles="""
#     button{
#         display: flex;
#         justify-content: flex-end;
#         width: 100%;
#     }
#     """
# ):
with col3:
    delete_images_button = st.button(
        label="Delete images",
        help="NOTE: This will remove all images that have been loaded to the site."
    )


# 1. Create the object
dm = DataModels()
print(dm)

if predict_button:
    # Use this only for prediction
    print("Training the model.")

    # Empty the faces directory if its already there
    if os.path.isdir("clustered_faces"):
        shutil.rmtree("clustered_faces")
        os.makedirs("clustered_faces")
    else:
        os.makedirs("clustered_faces")

    # 2. Generate the faces
    with st.spinner("Generating faces from images..."):
        dm.generate_faces_dnn(use_image_temp=True)

    # 3. Create the image embeddings for each face
    with st.spinner("Creating the image embeddings from each face..."):
        dm.tokenize()
        print("Final Result")
        print(dm.faces_vectors_x)

    # 5.2 Predict Clusters
    with st.spinner("Predicting labels from DBSCAN clustering model..."):
        print(f"Predictions: {dm.predict(st.session_state.dbscan_model)}")

    # 6. Create File tree for clusters
    with st.spinner("Creating file tree to group similar faces"):
        dm.group_faces()

    # 7. Output the final results (clusters) to streamlit
    # List all cluster folders in the directory
    clusters = [
        folder
        for folder in os.listdir(CLUSTER_DIR)
        if os.path.isdir(os.path.join(CLUSTER_DIR, folder))
    ]

    # Display each cluster
    for count, cluster in enumerate(clusters):
        if int(cluster[cluster.find("_") + 1 :]) == -1:
            st.header(f"Person: {count} (Outliers)")
        else:
            st.header(f"Person: {count}")

        # Get all images in this cluster
        cluster_folder = os.path.join(CLUSTER_DIR, cluster)
        faces = list(os.listdir(cluster_folder))

        # Display images in this cluster
        cols = st.columns(
            5
        )  # Adjust the number of columns based on how many images you want per row
        for idx, image_file in enumerate(faces):
            image_path = os.path.join(cluster_folder, image_file)
            img = Image.open(image_path)

            # Display image in a column
            with cols[idx % 5]:  # This ensures 5 images per row
                st.image(img, caption=image_file, use_column_width=True)

if train_button:
    # Use this to train the model and predict the images
    print("Training the model.")

    # Empty the faces directory if its already there
    if os.path.isdir("clustered_faces"):
        shutil.rmtree("clustered_faces")
        os.makedirs("clustered_faces")
    else:
        os.makedirs("clustered_faces")

    # 2. Generate the faces
    with st.spinner("Generating faces from images..."):
        dm.generate_faces_dnn()

    # 3. Create the image embeddings for each face
    with st.spinner("Creating the image embeddings from each face..."):
        dm.tokenize()
        print("Final Result")
        print(dm.faces_vectors_x)

    # 4. Plot the elbow plot for analysis
    with st.spinner("Plotting the elbow plot for analysis..."):
        min_samples = 3
        figure, eps = dm.plot_k_distance(min_samples)
        st.pyplot(figure)

    # 5.1 Generate DBSCAN
    with st.spinner("Generating DBSCAN clustering model for grouping similar faces"):
        dm.dbscan_model(eps, min_samples, use_knn=False)
        st.session_state.dbscan_model = dm.dbscan

    # 6. Create File tree for clusters
    with st.spinner("Creating file tree to group similar faces"):
        dm.group_faces()

    # 7. Output the final results (clusters) to streamlit
    # List all cluster folders in the directory
    clusters = [
        folder
        for folder in os.listdir(CLUSTER_DIR)
        if os.path.isdir(os.path.join(CLUSTER_DIR, folder))
    ]

    # Display each cluster
    for count, cluster in enumerate(clusters):
        if int(cluster[cluster.find("_") + 1 :]) == -1:
            st.header(f"Person: {count} (Outliers)")
        else:
            st.header(f"Person: {count}")

        # Get all images in this cluster
        cluster_folder = os.path.join(CLUSTER_DIR, cluster)
        faces = list(os.listdir(cluster_folder))

        # Display images in this cluster
        cols = st.columns(
            5
        )  # Adjust the number of columns based on how many images you want per row
        for idx, image_file in enumerate(faces):
            image_path = os.path.join(cluster_folder, image_file)
            img = Image.open(image_path)

            # Display image in a column
            with cols[idx % 5]:  # This ensures 5 images per row
                st.image(img, caption=image_file, use_column_width=True)

if delete_images_button:
    if os.path.isdir(IMAGES_DIR):
        shutil.rmtree(IMAGES_DIR)
        os.makedirs(IMAGES_DIR)
    else:
        os.makedirs(IMAGES_DIR)