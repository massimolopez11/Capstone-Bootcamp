"""Class for creatimg dataframes and models"""

import glob
import logging
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# from facenet_pytorch import InceptionResnetV1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataModels:
    """Class for dataframes and model manipulation"""

    def __init__(self) -> None:
        """Initialize a new data model consisting of a dataframe of images"""
        self.image_array = np.array(glob.glob("images/**/*.*", recursive=True))
        self.faces_array = (
            np.array(glob.glob("faces/images/**/*.*", recursive=True))
            if os.path.isdir("faces")
            else np.empty(0)
        )
        self.faces_vectors_x = np.empty(0)

    def tokenize(self, batch_size=100) -> None:
        """Tokenize the images"""
        # Initialize Img2Vec with GPU
        img2vec = Img2Vec(cuda=False)

        for i in range(0, self.faces_array.size, batch_size):
            faces_batch = self.faces_array[i : i + batch_size]

            # Open the images
            faces = [Image.open(name.replace("\\", "/")) for name in faces_batch]

            batch_vectors = img2vec.get_vec(faces)
            # Create the vector
            print(f"batch {i}: {batch_vectors.shape}")
            # for batch in batch_vectors:
            if self.faces_vectors_x.shape == (0,):
                self.faces_vectors_x = batch_vectors
            else:
                self.faces_vectors_x = np.vstack((self.faces_vectors_x, batch_vectors))

            # Close the images once done
            for image in faces:
                image.close()

        # self.faces_vectors = np.concatenate(self.faces_vectors, axis=0)

    def generate_faces(self) -> None:
        """Generate the faces from the photo_df"""

        # Empty the faces directory
        if os.path.isdir("faces"):
            shutil.rmtree("faces")
            os.makedirs("faces")
        else:
            os.makedirs("faces")

        # Setup the classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        for value in self.image_array:
            filename: str = value.replace("\\", "/")
            filename_no_ext = filename[: filename.find(".")]
            print(filename)

            # Detect the faces using grayscale images
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
            )

            # For each face, save them in their respective file
            for i, (x, y, w, h) in enumerate(faces):
                # Extract the face region (ROI: Region of Interest)
                face = img[y : y + h, x : x + w]

                # cv2.imshow("face", face)
                # cv2.waitKey(3000)

                if not os.path.isdir(f"faces/{filename_no_ext}"):
                    os.makedirs(f"faces/{filename_no_ext}")

                # Save the face as a new image
                face_filename = f"faces/{filename_no_ext}/face_{i+1}.jpg"
                cv2.imwrite(face_filename, face)
                print(f"Saved face {i+1} at {face_filename}")

        # Create an array for all the images in the faces directory
        self.faces_array = np.array(glob.glob("faces/images/**/*.*", recursive=True))

    def plot_k_distance(self, min_samples=3) -> plt.Figure:
        """Plot the k distance elbow plot for analysis"""
        # Compute the k-nearest neighbors
        nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors = nearest_neighbors.fit(self.faces_vectors_x)
        distances, indices = neighbors.kneighbors(self.faces_vectors_x)

        # Sort the distances and plot the k-distance graph
        distances = np.sort(
            distances[:, -1]
        )  # Get the k-th nearest neighbor distance for each point
        fig = plt.figure(figsize=(8, 8))

        plt.plot(distances)
        plt.ylabel("k-distance")
        plt.xlabel("Data points (sorted by distance)")
        plt.title("K-distance plot for DBSCAN")

        return fig

    def dbscan_model(self):
        """Generate DBSCAN clustering model"""
        self.dbscan = DBSCAN(eps=6.3, min_samples=3, metric="euclidean")
        self.clusters = self.dbscan.fit_predict(self.faces_vectors_x)
        print(len(set(self.clusters)))

        # Output cluster labels (-1 indicates noise/outliers, 0, 1, 2... are cluster labels)
        print("Cluster labels for each face:", self.clusters)

        # You can group the faces by their cluster labels
        # For example, group faces into dictionaries by cluster label:
        noise_num = 0
        not_noise_num = 0
        clustered_faces: dict = {}
        for i, label in enumerate(self.clusters):
            if label != -1:  # Ignore noise/outliers
                if label not in clustered_faces:
                    clustered_faces[label] = []
                clustered_faces[label].append(self.faces_vectors_x[i])
                print(f"({self.faces_array[i]}) Face {i+1} is Person {label}")
                not_noise_num += 1
            else:
                print(f"({self.faces_array[i]}) Face {i+1} identified as noise/outlier")
                noise_num += 1

        print(
            f"Found {len(set(self.clusters)) - (1 if -1 in self.clusters else 0)} clusters"
        )
        print(f"Ratio not noise to noise = {not_noise_num/(not_noise_num + noise_num)}")

    def group_faces(self):
        """Creates the file tree for the clusters for streamlit to display clusters."""
        # Path to the directory where you want to save the clusters
        output_dir = "clustered_faces"

        # Create the main output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create directories for each cluster and save the faces into the corresponding folders
        for idx, (face_image, label) in enumerate(
            zip([file for file in self.faces_array], self.clusters)
        ):
            # Create a folder for each cluster label
            cluster_folder = os.path.join(output_dir, f"cluster_{label}")

            # Create the folder if it doesn't exist
            if not os.path.exists(cluster_folder):
                os.makedirs(cluster_folder)

            # Get the image name (assuming you're working with file paths)
            image_name = os.path.basename(face_image)

            # Load the face image (if you're working with PIL images, use Image.save)
            img = Image.open(face_image)

            # Save the face image into the corresponding cluster folder
            img.save(os.path.join(cluster_folder, f"{image_name}"))

        print(f"Faces successfully grouped into {output_dir} based on clusters.")