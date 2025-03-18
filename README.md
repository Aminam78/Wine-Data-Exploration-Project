# Wine Data Exploration Project

This is a project for assignment 1 of the "Advanced Data Mining Course" at IUST University.
This project performs a comprehensive analysis of the Wine Dataset using Python. The goal is to explore the data, calculate feature correlations, and apply Principal Component Analysis (PCA) for dimensionality reduction and visualization.

## Project Description
- **Dataset**: The Wine Dataset is loaded from the `sklearn.datasets` library, containing 178 samples with 13 chemical features of wines and three target classes.
- **Steps Performed**:
  1. Loading and displaying initial dataset information.
  2. Exploring the dataset to identify the feature with the smallest and largest values and the largest range.
  3. Calculating the correlation matrix and identifying the strongest correlation.
  4. Applying PCA on unstandardized data and plotting the first two components.
  5. Standardizing the dataset and applying PCA on standardized data, then plotting the results.

## Prerequisites
- Python 3.7 or higher
- Required packages: `scikit-learn`, `numpy`, `pandas`, `matplotlib`

## How to Run
1. **Install Prerequisites**:
   - Create a virtual environment:
     ```bash
     python -m venv wine_project_env
          ```
   - Activate the virtual environment:
     - Windows: `wine_project_env\Scripts\activate`
     - Mac/Linux: `source wine_project_env/bin/activate`
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
2. **Run the Project**:
   - Navigate to the project directory:
     ```bash
     cd wine_project
     ```
   - Execute the main file:
     ```bash
     python src/main.py
     ```
3. **Output**:
   - Results will be displayed in the console.
   - PCA plots will be saved in the `output/` folder (as `unstandardized_pca.png` and `standardized_pca.png`).
  Project Structure
text

##Project Structure

```
wine_project/
├── src/                     # Source code
│   ├── main.py             # Main execution file
├── output/                  # Output plots
│   ├── unstandardized_pca.png
│   ├── standardized_pca.png
├── requirements.txt         # Dependency list
├── .gitignore               # Ignored files
├── README.md                # This file
```
##Author

Amirhossein Amin Moghaddam - Master's Student in Computer Software Engineering at Iran University of Science and Technology

Advanced Data Mining Course
March 2025
