import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

class DataSplitter:
    def split(self, dataframe):
        # Split the original training set into training and validation sets
        train_data, validation_data = train_test_split(
            dataframe,
            test_size=0.2,
            random_state=Config.RANDOM_SEED,
            stratify=dataframe[Config.LABEL_COLUMN]
        )

        print("\n========== DATA SPLITTING ==========")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(validation_data)}")

        return (
            train_data.reset_index(drop=True),
            validation_data.reset_index(drop=True)
        )