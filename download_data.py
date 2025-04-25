"""
This module provides a DataDownloader class for downloading and extracting image datasets.
"""

import os
import zipfile
from pathlib import Path
import requests

class DataDownloader:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataDownloader with a data directory.
        
        Args:
            data_dir (str): Directory where data will be stored
        """
        self.data_path = Path(data_dir)
        self.image_path = self.data_path / "pizza_steak_sushi"
        self.zip_path = self.data_path / "pizza_steak_sushi.zip"
        self.url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"

    def download_data(self) -> None:
        """
        Download the dataset if it doesn't exist locally.
        """
        if self.image_path.is_dir():
            print(f"{self.image_path} directory exists.")
            return

        print(f"Did not find {self.image_path} directory, creating one...")
        self.image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        print(f"Downloading pizza, steak, sushi data ...")
        req = requests.get(url=self.url)
        
        with open(self.zip_path, "wb") as f:
            f.write(req.content)

    def extract_data(self) -> None:
        """
        Extract the downloaded zip file.
        """
        if not self.zip_path.exists():
            print("No zip file found. Please download the data first.")
            return

        print("Unzipping pizza, steak, sushi data ...")
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.image_path)
        
        # Remove zip file
        os.remove(self.zip_path)

    def setup(self) -> None:
        """
        Complete setup process: download and extract data if needed.
        """
        self.download_data()
        self.extract_data()


if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.setup()
    

