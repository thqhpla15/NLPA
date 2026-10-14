import pandas as pd
from datasets import load_dataset

from config import Config

class DatasetLoader:
    def __init__(self):
        self.dataset = None

    def load(self):
        # Download the dataset on the first run and use cache afterward
        print("Loading IMDb dataset...")

        self.dataset = load_dataset(Config.DATASET_NAME)

        print("Dataset loaded successfully.")
        print(self.dataset)

        return self.dataset


    def get_dataframe(self, split="train"):
        # Convert a selected dataset split into a pandas DataFrame
        if self.dataset is None:
            raise RuntimeError("Dataset has not been loaded.")

        dataframe = self.dataset[split].to_pandas()

        # Keep only the columns required for sentiment classification
        dataframe = dataframe[
            [Config.TEXT_COLUMN, Config.LABEL_COLUMN]
        ].copy()

        return dataframe