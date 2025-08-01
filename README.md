# Crop Recommendation System Using Machine Learning
A machine learning-based system to recommend optimal crops based on soil, climate, and environmental conditions, aimed at helping farmers and agricultural professionals make better decisions for maximizing yields and profitability.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Experiment Results](#experiment-results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Dependencies](#Dependencies)

## Overview
This repository offers a machine learning pipeline to predict the most suitable crop based on specific environmental and soil properties. By leveraging advanced predictive models and historical data, the system delivers personalized crop recommendations tailored to the conditions of a given region or farm. Key factors considered include soil nutrient content (N, P, K), temperature, humidity, rainfall, and pH level.

## Dataset
The system uses a dataset augmented with rainfall, climate, and fertilizer data relevant to India. The key attributes are:
- **N:** Nitrogen in soil
- **P:** Phosphorous in soil
- **K:** Potassium in soil
- **Temperature:** (°C)
- **Humidity:** (%)
- **pH:** Soil pH
- **Rainfall:** (mm)

## Key Features
- **Input Data Collection:** Accepts user input for soil and environmental parameters.
- **Data Preprocessing:** Handles missing values and scales features with normalization.
- **Multiple ML Models:** Includes Decision Trees, Random Forests, SVM, and Gradient Boosting for accurate predictions.
- **Model Training and Evaluation:** Models are evaluated via relevant metrics to ensure reliability.
- **Crop Recommendation:** Suggests suitable crops for provided soil/climate input.

## Technologies Used

- **Python:** Backend and ML development
- **Scikit-learn:** Model building, training, evaluation
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations[2]

## Experiment Results
- **Outlier Analysis:** All columns except Nitrogen (N) have outliers
- **Train/Test Split:** 80% train, 20% validation
- **Top Performing Model:** Gaussian Naive Bayes (GaussianNB) with:
   - Training Accuracy: **93.26%**
   - Validation Accuracy: **92.53%**

## Installation

1. **Clone this repository:**
    ```
    git clone https://github.com/KRUTHIKTR/Crop-Recommendation-System-Using-Machine-Learning.git
    cd Crop-Recommendation-System-Using-Machine-Learning
    ```

2. **Create a virtual environment (optional):**
    ```
    python -m venv venv
    source venv/bin/activate        # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

## Usage
- Open the main Jupyter Notebook and follow instructions for exploratory data analysis and modeling.
- Input your environmental and soil parameters in the notebook or application.
- Run all cells to train models and receive crop recommendations.
- Evaluate the system using your own test data for accuracy estimation.

## Project Structure
```
Crop-Recommendation-System-Using-Machine-Learning/
├── data/
│ ├── crop_data.csv
├── Crop-Recommendation-Notebook.ipynb
├── requirements.txt
├── README.md
└── [output_and_model_files]
```
