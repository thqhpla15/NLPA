import matplotlib.pyplot as plt
import pandas as pd

from config import Config

class DatasetInspector:
    def inspect(self, dataframe: pd.DataFrame):
        # Display basic dataset information
        print("\n========== DATASET INFORMATION ==========")
        print(f"Number of reviews: {len(dataframe)}")
        print(f"Number of columns: {len(dataframe.columns)}")

        print("\nFirst 5 reviews:")
        print(dataframe.head())

        # Check missing values and duplicate reviews
        print("\nMissing values:")
        print(dataframe.isnull().sum())

        print(f"\nDuplicate reviews: {dataframe.duplicated().sum()}")

        # Show the number of reviews in each sentiment class
        print("\nClass distribution:")
        print(
            dataframe[Config.LABEL_COLUMN]
            .map(Config.LABEL_NAMES)
            .value_counts()
        )

        self.plot_class_distribution(dataframe)

        # Calculate document length in words
        dataframe["document_length"] = (dataframe["text"].fillna("").str.split().str.len())

        print("\nDocument length statistics:")
        print(dataframe["document_length"].describe())

        print("\nAverage document length by class:")
        print(dataframe.groupby("label")["document_length"].mean())

        # Calculate class distribution percentages
        class_counts = dataframe["label"].value_counts().sort_index()
        class_percentages = (dataframe["label"].value_counts(normalize=True).sort_index() * 100)

        print("\nClass distribution with percentages:")
        for label, count in class_counts.items():
            print(
                f"{Config.LABEL_NAMES[label]}: "
                f"{count} ({class_percentages[label]:.2f}%)"
            )


    def plot_class_distribution(self, dataframe: pd.DataFrame):
        # Count reviews belonging to each sentiment class
        class_counts = (
            dataframe[Config.LABEL_COLUMN]
            .map(Config.LABEL_NAMES)
            .value_counts()
            .reindex(["Negative", "Positive"], fill_value=0)
        )

        # Draw and save the class distribution chart
        plt.figure(figsize=(8, 5))
        class_counts.plot(kind="bar")

        plt.title("IMDb Dataset - Class Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Reviews")
        plt.xticks(rotation=0)
        plt.tight_layout()

        output_path = Config.OUTPUT_DIR / "class_distribution.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"\nClass distribution chart saved to: {output_path}")


