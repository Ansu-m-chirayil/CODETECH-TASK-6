Name: ANSU M CHIRAYIL 
Company: CODTECH IT SOLUTIONS
ID: CT08DA806
Domain: Data Analytics
Duration: June to August 2024
Mentor: SRAVANI GOUNI

Overview of the Project

# Project : FRAUD DETECTION IN FINANCIAL TRANSACTIONS ON CREDIT CARD DATASET FROM KAGGLE

# OBJECTIVE
Develop a robust fraud detection system leveraging machine learning techniques to identify fraudulent transactions within financial datasets. The system should efficiently detect anomalies and potentially fraudulent activities to mitigate financial losses and enhance security. Also after deploying different models, compare the accuracies to select the best fit model for future predictions.

# KEY ACTIVITIES
1. EXPLORATORY DATA ANALYSIS
2. HANDLING IMBALANCED DATA
3. TRAINING AND COMPARING THE MODEL

# LIBRARIES USED
1. **NumPy**: A fundamental package for scientific computing with Python, offering support for arrays, mathematical functions, and more.
2. **pandas**: Used for data manipulation and analysis, providing data structures like DataFrames.
3. **matplotlib.pyplot**: A module within Matplotlib employed for creating static, animated, and interactive visualizations.
4. **seaborn**: Utilized for making statistical graphics that are informative and attractive, built on top of Matplotlib.
5. **sklearn.model_selection.train_test_split**: A function for splitting data arrays into random train and test subsets.
6. **sklearn.preprocessing.StandardScaler**: A preprocessing module used for scaling features to have zero mean and unit variance.
7. **sklearn.metrics.classification_report**: A function that builds a text report showing the main classification metrics.
8. **sklearn.metrics.accuracy_score**: A function to compute the accuracy classification score.
9. **sklearn.linear_model.LogisticRegression**: A module to perform logistic regression, a linear model for binary classification.
10. **sklearn.metrics.confusion_matrix**: A function to compute a confusion matrix to evaluate the accuracy of a classification.
11. **pylab.rcParams**: A configuration module in Matplotlib used for customizing plot appearance globally.
12. **imblearn.combine.SMOTETomek**: A method that combines SMOTE (Synthetic Minority Over-sampling Technique) and Tomek links to handle imbalanced datasets by oversampling the minority class and undersampling the majority class.
13. **imblearn.under_sampling.NearMiss**: A technique for undersampling the majority class by selecting samples that are closest to the minority class.
14. **collections.Counter**: A container from the collections module that counts the occurrences of elements in a collection, such as a list or a tuple. 

# Key Insights
- **Classification Algorithms Evaluated**:
  - Logistic Regression
  - k-Nearest Neighbors (k-NN)
  - Decision Trees
  - Naive Bayes
  - Support Vector Machines (SVM)
- **Dataset and Preprocessing**:
  - Utilized the Kaggle credit card dataset for a binary classification problem.
  - Applied the SMOTETomek technique to handle class imbalance by combining oversampling of the minority class and undersampling of the majority class.
- **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
- **Performance Results**:
  - The **Decision Tree classifier** achieved the highest accuracy among the evaluated models.
  - The Decision Tree model's effectiveness is attributed to its ability to capture underlying patterns in the data and its interpretability.
- **Visual Comparison**:
  - A bar graph was used to visually compare the accuracy of each model, highlighting the superior performance of the Decision Tree classifier.

# Future Work
- **Advanced Techniques and Algorithms**:
  - Explore ensemble methods like Random Forest and Gradient Boosting.
  - Investigate the potential of neural networks for further performance improvement.
- **Hyperparameter Tuning**:
  - Fine-tuning the hyperparameters of each model could lead to better classification results.
- **Scalability and Robustness**:
  - Apply the models to more complex and larger datasets to evaluate their robustness.
  - Continual assessment with different datasets and extensive cross-validation techniques to build a more generalized and reliable classification model.
