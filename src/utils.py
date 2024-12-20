import os
import sys

import numpy  as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException



import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Perform GridSearchCV
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, scoring='accuracy')
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate classification metrics
            acc = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            conf_matrix = confusion_matrix(y_test, y_test_pred)

            # Ensure the confusion matrix has the correct shape
            if conf_matrix.shape == (2, 2):
                TN, FP, FN, TP = conf_matrix.ravel()
            else:
                TN, FP, FN, TP = 0, 0, 0, 0  # Default values if matrix is unexpected

            # Store results in the report
            report[list(models.keys())[i]] = {
                'Accuracy': acc,
                'Precision': precision,
                'Recall': recall,
                'False Positives': FP,
                'False Negatives': FN,
                'Best Parameters': gs.best_params_
            }

            # Print evaluation results
            print(f"Model: {list(models.keys())[i]}")
            print(f"Best Parameters: {gs.best_params_}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"False Positives: {FP}")
            print(f"False Negatives: {FN}")

            print("report", report)

        return report         
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)