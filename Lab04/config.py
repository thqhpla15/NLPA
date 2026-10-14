from pathlib import Path

class Config:
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = BASE_DIR / "output"

    # Dataset configuration
    DATASET_NAME = "stanfordnlp/imdb"
    TEXT_COLUMN = "text"
    LABEL_COLUMN = "label"

    # Label mapping
    LABEL_NAMES = {
        0: "Negative",
        1: "Positive"
    }

    # Random seed for reproducibility
    RANDOM_SEED = 42

    @classmethod
    def create_directories(cls):
        # Create output directory if it does not exist
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)