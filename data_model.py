"""Class for creatimg dataframes and models"""

import glob
import logging
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

# from facenet_pytorch import InceptionResnetV1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LAST_DBSCAN_MODEL: DBSCAN = None
"""The last dbscan model used. Object is stored for prediction"""

LAST_KNN_MODEL: KNeighborsClassifier = None
"""The last knn model used. Object is stored for prediction"""


class DataModels:
    """Class for dataframes and model manipulation"""

    def __init__(self) -> None:
        """Initialize a new data model consisting of a dataframe of images"""
        self.image_array = np.array(glob.glob("images/**/*.*", recursive=True))
        self.image_temp_array = np.array(
            glob.glob("images_temp/**/*.*", recursive=True)
        )
        self.faces_array = (
            np.array(glob.glob("faces/images/**/*.*", recursive=True))
            if os.path.isdir("faces")
            else np.empty(0)
        )
        self.faces_vectors_x = np.empty(0)

        self.dbscan: DBSCAN = None
        self.clusters = np.empty(0)

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

    def generate_faces(self, use_image_temp=False) -> None:
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

        for value in self.image_array if not use_image_temp else self.image_temp_array:
            filename: str = value.replace("\\", "/")
            filename_no_ext = filename[: filename.find(".")]
            print(filename)

            # Detect the faces using grayscale images
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100)
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
        self.faces_array = (
            np.array(glob.glob("faces/images/**/*.*", recursive=True))
            if not use_image_temp
            else np.array(glob.glob("faces/images_temp/**/*.*", recursive=True))
        )

    def plot_k_distance(self, min_samples) -> tuple[plt.Figure, float]:
        """Plot the k distance elbow plot for analysis"""
        # Compute the k-nearest neighbors
        nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors = nearest_neighbors.fit(self.faces_vectors_x)
        distances, _ = neighbors.kneighbors(self.faces_vectors_x)

        # Sort the distances and plot the k-distance graph
        distances = np.sort(
            distances[:, -1]
        )  # Get the k-th nearest neighbor distance for each point

        # Get the elbow point
        elbow_point_x, elbow_point_y = self._elbow_point(distances)

        fig = plt.figure(figsize=(8, 8))

        plt.plot(distances)
        plt.axvline(
            x=elbow_point_x,
            color="red",
            linestyle="--",
            label=f"Elbow point: {elbow_point_x}",
        )
        plt.axhline(
            y=elbow_point_y,
            color="green",
            linestyle="--",
            label=f"Elbow point: {elbow_point_y}",
        )
        plt.annotate(
            # text='({0:.2d}, {0:.2f})'.format(elbow_point_x, elbow_point_y),
            text=f"({elbow_point_x, elbow_point_y})",
            xy=(elbow_point_x, elbow_point_y),
        )
        plt.ylabel("k-distance")
        plt.xlabel("Data points (sorted by distance)")
        plt.title(f"K-distance plot for DBSCAN (Min Samples = {min_samples})")

        return (fig, elbow_point_y)

    def _elbow_point(self, distances: np.ndarray) -> int:
        """Find the index of the elbow point using the maximum curvature method."""
        # Get the total number of points
        num_points = len(distances)

        # Create a line from the first point to the last point
        start_point = np.array([0, distances[0]])
        end_point = np.array([num_points - 1, distances[-1]])

        # Create a vector representing the line
        line_vec = end_point - start_point

        # Compute the distance of each point from the line
        distances_to_line = []
        for i in range(num_points):
            point = np.array([i, distances[i]])
            vec_from_start = point - start_point
            distance = np.linalg.norm(
                np.cross(line_vec, vec_from_start)
            ) / np.linalg.norm(line_vec)
            distances_to_line.append(distance)

        # Find the point with the maximum distance to the line (the elbow point)
        elbow_point = np.argmax(distances_to_line)
        print(elbow_point, distances[elbow_point])

        return (elbow_point, distances[elbow_point])

    def dbscan_model(self, eps: float, min_samples):
        """Generate DBSCAN clustering model"""
        print(f"Using esp and min sample values of ({eps, min_samples})")

        self.dbscan = DBSCAN(eps=eps - 2, min_samples=min_samples, metric="euclidean")
        self.clusters = self.dbscan.fit_predict(self.faces_vectors_x)

        print(len(set(self.clusters)))

        # Output cluster labels (-1 indicates noise/outliers))
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
                # print(f"({self.faces_array[i]}) Face {i+1} is Person {label}")
                not_noise_num += 1
            else:
                # print(f"({self.faces_array[i]}) Face {i+1} identified as noise/outlier")
                noise_num += 1

        print(
            f"Found {len(set(self.clusters)) - (1 if -1 in self.clusters else 0)} clusters"
        )
        print(
            f"Ratio of outliers = {not_noise_num/(not_noise_num + noise_num)} \
                  ({not_noise_num}/{(not_noise_num + noise_num)})"
        )


    def predict(self, dbscan_model: DBSCAN):
        """Only used for predicting the model. Note that the model has to be set
        up or else the function will naturally throw an error.
        """
        eps: float = dbscan_model.eps

        # Nearest neighbors based on the core points
        neighbors = NearestNeighbors(radius=eps)
        neighbors.fit(dbscan_model.components_)

        # Find the nearest core point for each new point
        distances, indices = neighbors.radius_neighbors(self.faces_vectors_x)

        # Create an array to store predicted labels
        predictions = np.full(
            self.faces_vectors_x.shape[0], -1
        )  # Default to noise (-1)

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if len(idx) > 0:  # If there are neighbors within eps
                # Assign the label of the nearest core sample
                predictions[i] = dbscan_model.labels_[
                    dbscan_model.core_sample_indices_[idx[0]]
                ]

        self.clusters = predictions
        return predictions

    def group_faces(self):
        """Creates the file tree for the clusters for streamlit to display clusters."""
        # Path to the directory where you want to save the clusters
        output_dir = "clustered_faces"

        # Create the main output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create directories for each cluster and save the faces into the corresponding folders
        for face_image, label in zip(list(self.faces_array), self.clusters):
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
