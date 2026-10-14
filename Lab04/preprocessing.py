import re
import pandas as pd
from config import Config


class TextPreprocessor:
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"<br\s*/?>", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def preprocess_dataframe(self, dataframe):
        dataframe = dataframe.copy()

        dataframe[Config.TEXT_COLUMN] = (
            dataframe[Config.TEXT_COLUMN]
            .fillna("")
            .apply(self.clean_text)
        )

        dataframe = dataframe.dropna(
            subset=[Config.LABEL_COLUMN]
        )

        return dataframe.reset_index(drop=True)