import pandas as pd
import logging
import os

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataProcessor:
    def __init__(self):
        self.dataframe = None

    def load_dataset(self, file_path):
        """📂 Loads a dataset from a CSV file."""
        try:
            if not os.path.exists(file_path):
                return "❌ Error: File not found!"
                
            self.dataframe = pd.read_csv(file_path)
            logging.info(f"✅ Dataset loaded successfully: {file_path}")

            return f"✅ Dataset loaded successfully with {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns."

        except Exception as e:
            logging.error(f"❌ Error loading dataset: {str(e)}")
            return f"❌ Error loading dataset: {str(e)}"

    def get_columns(self):
        """📊 Returns the column names of the dataset."""
        if self.dataframe is not None:
            return list(self.dataframe.columns)
        return "❌ No dataset loaded."

    def get_summary(self):
        """📈 Returns basic statistics of the dataset."""
        if self.dataframe is not None:
            return self.dataframe.describe().to_dict()
        return "❌ No dataset loaded."

    def handle_missing_values(self, strategy="drop"):
        """🛠 Handles missing values using different strategies: 'drop', 'mean', or 'median'."""
        if self.dataframe is None:
            return "❌ No dataset loaded."

        try:
            if strategy == "drop":
                self.dataframe.dropna(inplace=True)
                logging.info("✅ Missing values dropped.")
                return "✅ Missing values dropped."
                
            elif strategy == "mean":
                self.dataframe.fillna(self.dataframe.mean(), inplace=True)
                logging.info("✅ Missing values filled with mean.")
                return "✅ Missing values filled with mean."

            elif strategy == "median":
                self.dataframe.fillna(self.dataframe.median(), inplace=True)
                logging.info("✅ Missing values filled with median.")
                return "✅ Missing values filled with median."

            else:
                return "❌ Invalid strategy! Use 'drop', 'mean', or 'median'."
        except Exception as e:
            logging.error(f"❌ Error handling missing values: {str(e)}")
            return f"❌ Error handling missing values: {str(e)}"

    def get_data_head(self, rows=5):
        """📌 Returns the first few rows of the dataset."""
        if self.dataframe is not None:
            return self.dataframe.head(rows).to_dict()
        return "❌ No dataset loaded."

    def save_cleaned_data(self, output_path="cleaned_data.csv"):
        """💾 Saves the cleaned dataset to a new file."""
        if self.dataframe is not None:
            try:
                self.dataframe.to_csv(output_path, index=False)
                logging.info(f"✅ Cleaned dataset saved: {output_path}")
                return f"✅ Cleaned dataset saved to {output_path}."
            except Exception as e:
                logging.error(f"❌ Error saving dataset: {str(e)}")
                return f"❌ Error saving dataset: {str(e)}"
        return "❌ No dataset loaded."

# ✅ Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Load a dataset
    dataset_path = "sample_data.csv"
    print(processor.load_dataset(dataset_path))
    
    # Display columns
    print("📝 Columns:", processor.get_columns())

    # Handle missing values (choose: 'drop', 'mean', or 'median')
    print(processor.handle_missing_values(strategy="mean"))

    # Show summary stats
    print("📊 Summary:", processor.get_summary())

    # Show first 5 rows
    print("🔍 Head:", processor.get_data_head())

    # Save cleaned dataset
    cleaned_path = "cleaned_sample_data.csv"
    print(processor.save_cleaned_data(cleaned_path))
