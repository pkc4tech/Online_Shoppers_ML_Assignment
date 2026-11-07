# Online Shoppers Purchasing Intention Prediction

**Machine Learning — Assignment 2 | Classification and Streamlit Deployment**

This project compares five machine learning classification algorithms to predict whether an online shopping session will result in a purchase. The trained models are evaluated using the same held-out test dataset, and the results can be viewed through an interactive Streamlit application.

## a. Problem statement

An e-commerce website receives visitors who browse products, review pages, and sometimes leave without buying anything. The objective of this project is to predict whether a browsing session will lead to a purchase, using information about visitor behavior and browsing activity.

This is a **binary classification** problem:

- `Revenue = True` (1): a purchase occurred.
- `Revenue = False` (0): no purchase occurred.

The project implements Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM). Their performance is compared using accuracy, AUC, precision, recall, F1 score, and Matthews Correlation Coefficient (MCC).

## b. Dataset description

- **Dataset:** Online Shoppers Purchasing Intention Dataset
- **Kaggle source:** https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset
- **Original source:** UCI Machine Learning Repository
- **Dataset file:** `online_shoppers_intention.csv`
- **Total records:** 12,330 browsing sessions
- **Input features:** 17
- **Target column:** `Revenue`
- **Target distribution:** 10,422 non-purchase sessions (84.53%) and 1,908 purchase sessions (15.47%)
- **Missing values:** None in the supplied dataset
- **Exact duplicate records:** 125, retained in this experiment

The input features describe browsing activity and visitor information, including the number and duration of administrative, informational, and product-related page visits; bounce and exit rates; page values; special day proximity; month; operating system; browser; region; traffic type; visitor type; and weekend indicator.

### Data preparation

The data is split into **80% training (9,864 sessions)** and **20% testing (2,466 sessions)** using a stratified split with `random_state=42`. Stratification preserves the purchase/non-purchase proportions across the two sets.

The preprocessing pipeline standardizes ten continuous attributes and one-hot encodes seven categorical or identifier attributes. Preprocessing is fitted using the training set only, and then applied to the test set. The saved `test_data.csv` includes all 17 input features and the `Revenue` target.

Since only about 15.5% of sessions end in purchases, accuracy alone is not enough to judge model quality. F1 score, recall, precision, AUC, and MCC are also considered. No class resampling is used in this experiment.

## c. GitHub repository link

**GitHub repository:** https://github.com/pkc4tech/Online_Shoppers_ML_Assignment

## d. Models used and evaluation results

All five models are trained on the same training split and evaluated on the same **2,466-session test set**. The table below contains the results displayed by the Streamlit application.

| ML Model Name                |   Accuracy |        AUC | Precision |     Recall |         F1 |        MCC |
| ---------------------------- | ---------: | ---------: | --------: | ---------: | ---------: | ---------: |
| Logistic Regression          |     0.8808 |     0.8828 |    0.7366 |     0.3586 |     0.4824 |     0.4592 |
| Decision Tree                | **0.8958** | **0.9137** |    0.7273 |     0.5236 | **0.6088** | **0.5604** |
| KNN                          |     0.8731 |     0.8164 |    0.6927 |     0.3246 |     0.4421 |     0.4159 |
| Naive Bayes (Gaussian)       |     0.2729 |     0.7334 |    0.1726 | **0.9738** |     0.2933 |     0.1289 |
| Support Vector Machine (SVM) |     0.8873 |     0.8824 |    0.7047 |     0.4686 |     0.5629 |     0.5149 |

For precision, recall, and F1, the positive class is **purchase (`Revenue=True`)**. AUC is calculated using probability scores or, for SVM, decision scores.

### Observations about model performance

| ML Model Name          | Observation about model performance                                                                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression    | Accuracy is 88.08%, but recall is only 35.86%. The model correctly identifies many non-purchases but misses a large number of actual purchases.                        |
| Decision Tree          | It has the highest accuracy (89.58%), AUC (0.9137), F1 (0.6088), and MCC (0.5604) among the five models. It gives the best overall result in this experiment.          |
| KNN                    | Accuracy is 87.31%, but recall is relatively low (32.46%). It performs below Decision Tree and SVM on identifying purchase sessions.                                   |
| Naive Bayes            | Recall is very high (97.38%), but precision is only 17.26%. It predicts many sessions as purchases even when no purchase occurred, resulting in poor overall accuracy. |
| Support Vector Machine | Accuracy is 88.73% and F1 is 0.5629. It performs better than Logistic Regression and KNN on purchase-class F1, but does not exceed Decision Tree.                      |
| **Overall winner**     | **Decision Tree** — it has the strongest combination of F1, MCC, AUC, and accuracy on this test set.                                                                   |

### Conclusion

Based on the observed results, **Decision Tree is the best-performing model for this dataset in the current experiment**. Although Logistic Regression and SVM also achieve relatively high accuracy, their recall for actual purchases is lower. Naive Bayes identifies most purchases but produces too many false positives. The comparison shows why multiple evaluation metrics are useful when the target classes are imbalanced.

These findings apply to this particular train/test split and model configuration. Cross-validation, hyperparameter tuning on training data, and checking the effect of duplicate records could provide a more robust comparison in future work.

## Project files

```text
Online_Shoppers_ML_Assignment/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── online_shoppers_intention.csv
├── test_data.csv
├── metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metadata.json
```

The `.venv` virtual environment is not included in the repository.

## How to run the project

Use a recent supported Python version and run the following commands from the project directory.

**1. Create a virtual environment**

```bash
python -m venv .venv
```

**2. Install the dependencies**

On Windows, this command can be run without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**3. Train and evaluate the models (optional if saved models are already present)**

```powershell
.\.venv\Scripts\python.exe train_models.py
```

**4. Start the Streamlit app**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in the browser. Upload the provided **`test_data.csv`**, choose a classifier from the dropdown, and review the six evaluation metrics, confusion matrix, and classification report. The application loads saved trained models and **does not retrain them** when test data is uploaded. It also displays a comparison table for all five models.

## Streamlit application link

**Live app:** https://onlineshoppersmlassignment-4ersynnidfqvnyagflualn.streamlit.app/

To deploy, connect the GitHub repository to [Streamlit Community Cloud](https://share.streamlit.io/), select the `main` branch and `app.py` as the entry point, then verify the application using `test_data.csv`.
