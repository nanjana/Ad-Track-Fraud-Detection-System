# Ad-Track Fraud Detection System

## Overview
The **Ad-Track Fraud Detection System** is a machine learning project designed to detect fraudulent activities in ad tracking data. By leveraging a combination of feature engineering and machine learning models, this system can identify patterns in ad-click data to flag suspicious activities.

## Features
- **Data Preprocessing**: Cleans and preprocesses raw ad-tracking data.
- **Feature Engineering**: Extracts useful features like IP address, click time, and app/device information to enhance model performance.
- **Fraud Detection**: Implements machine learning models such as XGBoost to classify clicks as fraudulent or legitimate.
- **Model Evaluation**: Provides accuracy, AUC-ROC curve, and other metrics to evaluate the model's performance.

## Prerequisites
- Python 3.x

## How to Run the Project
   Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Ad-Track-Fraud-Detection-System.git  
   ```
## Navigate to the project directory:
   cd Ad-Track-Fraud-Detection-System

## Run the fraud detection script:
   python Code_Source.py

## Model Description
This system primarily uses the XGBoost algorithm for classification. The dataset includes features such as:

IP Address
Click Time
App ID
Device Type
Operating System
Channel

The goal is to classify whether each click is fraudulent.

## Results
The model is evaluated based on:
Accuracy
Precision
Recall
F1-Score
AUC-ROC


## Source Code:

