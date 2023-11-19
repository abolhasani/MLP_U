#!/bin/bash

echo "Step 1: Initiate preprocessing"
python3 Prep.py

sleep 1

echo "Step 2: Get pie charts for data distibution before SMOTE"
python3 pieunbalance.py

sleep 1

echo "Step 3: Apply SMOTE upsampling to make the distribution of classes similar"
python3 smote.py

sleep 1

echo "Step 4: Get pie charts for data distibution after SMOTE"
python3 pieunbalance.py

sleep 1

echo "Step 5: Plot feature importance of dataset features"
python3 barplot.py

sleep 1

echo "Step 6: Apply Hot enocder for converting categorical features into binary ones, suitable for linear models"
python3 HotEncode.py

sleep 1

echo "Step 7: Benchmarking with SKLearn's Bagged Trees"
python3 BAGTest.py

sleep 1

echo "Step 8: Benchmarking with SKLearn's Decision Trees"
python3 DTTest.py

sleep 1

echo "Step 9: Benchmarking with SKLearn's K-Nearest Neighbours"
python3 KNNTest.py

sleep 1

echo "Step 10: Benchmarking with LightGBM"
python3 LBGMTest.py

sleep 1

echo "Step 11: Benchmarking with SKLearn's Log Regression"
python3 LOGREGTest.py

sleep 1

echo "Step 12: Benchmarking with SKLearn's Multi-Layer Perceptron"
python3 MLPCTest.py

sleep 1

echo "Step 12: Benchmarking with SKLearn's Perceptron"
python3 PRCPTRNTest.py

sleep 1

echo "Step 13: Benchmarking with SKLearn's Random Forests"
python3 RFTest.py

sleep 1

echo "Step 14: Benchmarking with SKLearn's Stochastic Gradient Descent"
python3 SGDTest.py

sleep 1

echo "Step 15: Benchmarking with SKLearn's Support Vector Machines"
python3 SVMTest.py

sleep 1

echo "Step 16: Benchmarking with SKLearn's XGBoost"
python3 XGTest.py

sleep 1

echo "Step 17: Main Testing Model - Bagged Trees"
python3 BAGMain.py

sleep 1

echo "Step 18: Main Testing Model - Batch Gradient Descent (not included in the final report because of irrelevant results)"
python3 BGDMain.py

sleep 1

echo "Step 19: Main Testing Model - Decision Trees"
python3 DTMain.py

sleep 1

echo "Step 20: Main Testing Model - Logistic Regression"
python3 DTMain.py

sleep 1

echo "Step 21: Main Testing Model - Standard and Average Perceptron"
python3 PRCPTRNMain.py

sleep 1

echo "Step 22: Main Testing Model - Random Forest with 5-Fold Cross Validation"
python3 RFMain.py

sleep 1

echo "Step 23: Main Testing Model - Stochastic Gradient Descent"
python3 SGDMain.py

sleep 1

echo "Step 24 and Final: Main Testing Model - Support Vector Machines with 5-Fold Cross Validation"
python3 SVMMain.py
