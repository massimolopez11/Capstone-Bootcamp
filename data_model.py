"""Class for creatimg dataframes and models"""

import glob

import pandas as pd


class DataModels:
    """Class for dataframes and model manipulation"""

    def __init__(self) -> None:
        """Initialize a new data model consisting of a dataframe of images"""
        self.photo_df = pd.DataFrame.from_records(
            [glob.glob("images/**/*.*", recursive=True)]
        ).T
