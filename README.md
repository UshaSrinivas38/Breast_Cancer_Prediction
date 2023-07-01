# Breast_Cancer_Prediction
# Breast Cancer Prediction

This program is designed to predict breast cancer diagnosis using various machine learning algorithms such as Decision Tree, Random Forest, SVM, and Logistic Regression. It analyzes a given set of features extracted from breast cancer data and provides accurate predictions.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Evaluation](#models-and-evaluation)
- [Streamlit Deployment](#streamlit-deployment)


## Introduction

The Breast Cancer Prediction program utilizes a dataset containing features extracted from breast cancer samples. The goal is to accurately predict the diagnosis (malignant or benign) based on these features. The program employs various machine learning algorithms and provides evaluation metrics to assess the performance of each model.

## Installation

1. Clone the repository or download the source code files.

2. Install the required dependencies using the following command:
pip install -r requirements.txt

3. Download the breast cancer dataset (cancer_data.csv) and place it in the same directory as the program files.

## Usage

1. Run the program by executing the following command:
python main.py

2. After running the program, a Streamlit web application will open in your default browser.

3. Enter the values for the various features extracted from breast cancer samples.

4. Click the "Test Results" button to get the predicted diagnosis based on the input features.

## Models and Evaluation

The program utilizes four different machine learning models for breast cancer prediction: Decision Tree, Random Forest, SVM, and Logistic Regression.

For each model, the program performs the following steps:

1. Load the breast cancer dataset and preprocess it by dropping any missing values.

2. Convert the diagnosis labels (M and B) into numerical values (1 and 0).

3. Split the dataset into training and testing sets using an 80-20 split.

4. Train the model using the training data.

5. Evaluate the model using various metrics such as accuracy, precision, sensitivity, specificity, and AUC ROC.

6. Provide the evaluation metrics for each model.

7. Generate a bar graph showing the accuracy of each model.

8. Plot the confusion matrix for each model to visualize the performance.

9. Display the classification report for each model.

## Streamlit Deployment

The program includes a Streamlit web application for easy interaction with the breast cancer prediction model. It allows users to input the various features extracted from a breast cancer sample and get an instant diagnosis prediction.





