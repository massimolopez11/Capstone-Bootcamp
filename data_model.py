"""Class for creatimg dataframes and models"""

import glob

import pandas as pd

from img2vec_pytorch import Img2Vec
from PIL import Image


class DataModels:
    """Class for dataframes and model manipulation"""

    def __init__(self) -> None:
        """Initialize a new data model consisting of a dataframe of images"""
        self.photo_df = pd.DataFrame.from_records(
            [glob.glob("images/**/*.*", recursive=True)]
        ).T

        self.tokenize()

    def tokenize(self) -> None:
        """Tokenize the images"""
        # Initialize Img2Vec with GPU
        img2vec = Img2Vec(cuda=True)

        # Open the images
        images_list = [Image.open(name[0]) for name in self.photo_df.values]

        # Create the vector
        self.image_vectors = pd.DataFrame(img2vec.get_vec(images_list))

        # Close the images once done
        for image in images_list:
            image.close()

    def display_attributes(self) -> None:
        """Helper method to print out attributes."""
        print(self.photo_df, self.image_vectors)
