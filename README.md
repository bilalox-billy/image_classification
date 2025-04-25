# Image Classification Project

A Python project for image classification using PyTorch.

## Setup

1. Make sure you have Poetry installed. If not, install it using:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone this repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd image-classification
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Activate the virtual environment:

   **Windows (PowerShell):**
   ```powershell
   poetry shell
   ```

   **Windows (Command Prompt):**
   ```cmd
   poetry shell
   ```

   **Linux/macOS:**
   ```bash
   source $(poetry env info --path)/bin/activate
   ```

   **Alternative for Linux/macOS:**
   ```bash
   poetry shell
   ```

## Usage

The project includes a `DataDownloader` class for downloading and preparing the image dataset:

```python
from download_data import DataDownloader

# Create an instance
downloader = DataDownloader()

# Download and extract data
downloader.setup()
```

## Development

- Run tests:
  ```bash
  poetry run pytest
  ```

- Format code:
  ```bash
  poetry run black .
  poetry run isort .
  ```

- Check code quality:
  ```bash
  poetry run flake8
  ```

## Project Structure

```
image-classification/
├── data/                    # Data directory
├── download_data.py         # Data downloader module
├── pyproject.toml           # Poetry configuration
└── README.md               # This file
```