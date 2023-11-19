# Predicting Hospital Readmissions and Stay Duration for Diabetes Patients

This repository contains all the necessary files to run the project aimed to fullfill the requirements of CS6350 at The University Of Utah - Fall 2023

Project team: Sadegh Abolhasani, Hamid Firoozfar

Below, you can find the list of the files that exist in the repository, and a brief description of them. In the codes, we have done our best to unsure enough commenting has been done.

1. diabetic_data.csv 
- This is the main dataset file, we perform preprocessing activities on it and derive 5 other csv files, which are used by our implemented algorithms to give us the results.
- Preprocessed datasets are not included because of their huge size, but they will be created each time when you follow the execution steps. 

2. run.sh
- This is the bash file that runs all the codes that produce an output in this project.

3. utils.py
- Includes our manually implemented k-fold split and train-test split functions so we can ignor SKLearn ones.
- Usage: (provide k for k-fold) manual_k_fold_split(X, y, k=5) 
- Usage: (provide test size split percentage) custom_train_test_split(X, y, test_size=0.3)

4. Metrics.py
- Include the implementation for using SKLearn to get test accuracy, recall, r-squared, AUC (if applicable), precision, and F-1 score
- Only works for decision trees and its ensembles
- Usage: metrics_report(tree, test_data)

5. VMetrics.py
- same as Metrics,py, but for linear models
- Usage: metricsSV(y_true, y_pred)

6. DecisionTree.py
- The library file to implement necessary functions for decision tree and its ensembles. 
- It has been previously developed for the homework 1 and 2
- you can check it out here: https://github.com/abolhasani/ML_U/tree/main/EnsembleLearning

7. svm.py
- Includes the class and functions required for implementing SVM. 
- Previously implemented in homework 4
- You can check it out here and the readme file specifically for it: https://github.com/abolhasani/ML_U/tree/main/SVM

8. RF.py
- implementation of special random forest functions necessary to run the random forest codes, based on ID3 from DecisionTree (discussed before)

9. perceptron.py
- Includes all the functions neeeded for implementing perceptron (standard, voted, average)
- Previously developed for homework 3
- You can check it out here and the readme file specifically for it: https://github.com/abolhasani/ML_U/tree/main/Perceptron

# Project Execution Steps

## Preprocessing and Analysis

1. **Initiate preprocessing**
   - Run: `python3 Prep.py`
   - Description: Initial data preprocessing step.

2. **Get pie charts for data distribution before SMOTE**
   - Run: `python3 pieunbalance.py`
   - Description: Generates pie charts to visualize data distribution prior to SMOTE upsampling.

3. **Apply SMOTE upsampling to make the distribution of classes similar**
   - Run: `python3 smote.py`
   - Description: Applies SMOTE technique to balance the class distribution.

4. **Get pie charts for data distribution after SMOTE**
   - Run: `python3 pieunbalance.py`
   - Description: Generates pie charts to visualize data distribution after SMOTE upsampling.

5. **Plot feature importance of dataset features**
   - Run: `python3 barplot.py`
   - Description: Creates bar plots to display the importance of various dataset features.

6. **Apply Hot encoder for converting categorical features into binary ones, suitable for linear models**
   - Run: `python3 HotEncode.py`
   - Description: Converts categorical features into a binary format using hot encoding, facilitating linear model analysis.

## Benchmarking with SKLearn Models

7. **Benchmarking with SKLearn's Bagged Trees**
   - Run: `python3 BAGTest.py`
   - Description: Benchmark test using SKLearn's Bagged Trees model.

8. **Benchmarking with SKLearn's Decision Trees**
   - Run: `python3 DTTest.py`
   - Description: Benchmark test using SKLearn's Decision Trees model.

9. **Benchmarking with SKLearn's K-Nearest Neighbours**
   - Run: `python3 KNNTest.py`
   - Description: Benchmark test using SKLearn's K-Nearest Neighbours model.

10. **Benchmarking with LightGBM**
    - Run: `python3 LBGMTest.py`
    - Description: Benchmark test using LightGBM.

11. **Benchmarking with SKLearn's Logistic Regression**
    - Run: `python3 LOGREGTest.py`
    - Description: Benchmark test using SKLearn's Logistic Regression model.

12. **Benchmarking with SKLearn's Multi-Layer Perceptron**
    - Run: `python3 MLPCTest.py`
    - Description: Benchmark test using SKLearn's Multi-Layer Perceptron model.

13. **Benchmarking with SKLearn's Perceptron**
    - Run: `python3 PRCPTRNTest.py`
    - Description: Benchmark test using SKLearn's Perceptron model.

14. **Benchmarking with SKLearn's Random Forests**
    - Run: `python3 RFTest.py`
    - Description: Benchmark test using SKLearn's Random Forests model.

15. **Benchmarking with SKLearn's Stochastic Gradient Descent**
    - Run: `python3 SGDTest.py`
    - Description: Benchmark test using SKLearn's Stochastic Gradient Descent model.

16. **Benchmarking with SKLearn's Support Vector Machines**
    - Run: `python3 SVMTest.py`
    - Description: Benchmark test using SKLearn's Support Vector Machines model.

17. **Benchmarking with SKLearn's XGBoost**
    - Run: `python3 XGTest.py`
    - Description: Benchmark test using SKLearn's XGBoost model.

## Main Testing Models

18. **Main Testing Model - Bagged Trees**
    - Run: `python3 BAGMain.py`
    - Description: Main testing using the Bagged Trees model.

19. **Main Testing Model - Batch Gradient Descent**
    - Run: `python3 BGDMain.py`
    - Description: Main testing using the Batch Gradient Descent model. (Not included in the final report due to irrelevant results)

20. **Main Testing Model - Decision Trees**
    - Run: `python3 DTMain.py`
    - Description: Main testing using the Decision Trees model.

21. **Main Testing Model - Logistic Regression**
    - Run: `python3 DTMain.py`
    - Description: Main testing using the Logistic Regression model.

22. **Main Testing Model - Standard and Average Perceptron**
    - Run: `python3 PRCPTRNMain.py`
    - Description: Main testing using Standard and Average Perceptron models.

23. **Main Testing Model - Random Forest with 5-Fold Cross Validation**
    - Run: `python3 RFMain.py`
    - Description: Main testing using the Random Forest model with 5-fold cross-validation.

24. **Main Testing Model - Stochastic Gradient Descent**
    - Run: `python3 SGDMain.py`
    - Description: Main testing using the Stochastic Gradient Descent model.

25. **Main Testing Model - Support Vector Machines with 5-Fold Cross Validation**
    - Run: `python3 SVMMain.py`
    - Description: Main testing using the Support Vector Machines model with 5-fold cross-validation.
