# Correlation Analysis and Prediction of Cancer Risk based on HPV Status, Alcohol, and Tobacco Usage



## Project Description
This project aims to uncover correlations between the HPV status of a patient combined with their alcohol and tobacco usage to the possibility of cancer. The model attempts to predict these labels with many additional features.


### Filtering Scripts
This project additionally has a universal filtering script for all HN_P2C files(as csvs). 


## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Modeling](#modeling)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction
Understanding the risk factors for cancer is crucial for early detection and prevention. This project focuses on analyzing the relationship between HPV status, alcohol consumption, and tobacco usage with cancer risk. By incorporating various additional features, we aim to build a predictive model to assess the likelihood of cancer in patients.

## Dataset
We use a modified version of the HN_P2C dataset, which includes information on patients' HPV status, alcohol and tobacco usage, and other relevant coefficients related to the patients data. In order to use this script, you must already be provided with a .csv file for this case. If you have a .mat file, please convert it to .csv in the MATLAB IDE before utilizing it in the script.

## Features
The features considered in this project include, but are not limited to:
- HPV Status
- Alcohol Usage
- Tobacco Usage
- Age
- Gender
- Medical History
- Lifestyle Factors

## Modeling
The project involves several steps to build the predictive model:
1. Data Preprocessing: Cleaning and preparing the dataset for analysis.
2. Feature Engineering: Creating additional features that may enhance the predictive power of the model.
3. Model Selection: Evaluating different machine learning algorithms to determine the best performing model.
4. Model Training: Training the selected model on the dataset.
5. Model Evaluation: Assessing the model's performance using various metrics.


## Installation
To run this project locally, follow these steps:
1. Clone the repository:
   ```
   git clone https://github.com/marculab/Model-Interpretation
2. Navigate to the project directory:

```
cd Model-Interpretation #should be in here by default
```

3. Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
4. Install the required dependencies:
```
pip install -r requirements.txt
```
## Usage
To use the general filtering mechanism:

1. Navigate to General-Filtering/ and paste the csv file that you would like to simplify into the script as directed.

2. Run the script which will automatically create a directory and store the updated file into a HN_P2C/ folder.

### ML Usage:
To use the ML labelling filter:
3. Complete steps 1 and 2.
4. Navigate to the Ml_labelling.py file located in the Ayush-Modelling/ directory.
5. Paste the path to the filtered csv file provided prior, and run the script to obtain the ml_labelled data.


## Results

### Diagrams

1. **PaCMAP**
After running PaCMAP on the raw data, we were able to discern that the clustering was accurate, but needed additional balancing due to the lack of "Cancer" points on the table.
![pacmap_visualization_without_balancing](https://github.com/user-attachments/assets/f3f1f8fd-0abb-4d3d-b7b1-767d1de640c6)

2. **Initial Model**
After Balancing the model with One-Hot Encoding with PCA, additionally with SMOTE balancing, comparing the models suggests to us that the K-Neighbors Classifier is the best model in terms of accurately predicting labels without overcompensating with the biased amount of cancer datapoints.
![image](https://github.com/user-attachments/assets/77a1b809-0408-49d9-860f-8d05b0b3840a)

![image](https://github.com/user-attachments/assets/dae2165c-7c69-4ef9-b949-8dfca73ea5c5)

This table demonstrates the need to run a balancing technique on the dataset, as seen in the next table which was evaluated prior to SMOTE:
![image](https://github.com/user-attachments/assets/3e5e0c80-6511-4fd2-8678-b0490fa531ef)

The high accuracy for all the different types of models suggests that it was overfit to the cancer datapoints, which would give it a correct labelling the majority of the time.



#Thigns to do
Write down Patients excluded due to NA data

Leave One Patient Out Cross Validation
- See which patients are performing badly
- AUC, Sensitivity and Specificity for each patient
