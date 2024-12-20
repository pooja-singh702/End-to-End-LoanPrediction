import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


import pickle  # Importing pickle for saving models
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models




import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
# Target variable
#random_state = 34

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()



    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {"Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(class_weight={0: 1, 1: 100}),
            "SVM" : SVC(),
           "Gradient Boosting": GradientBoostingClassifier()                        
                      
          }
            params = { "Logistic Regression": {'C': [0.001, 0.01, 0.1, 1, 10],
                                               'penalty': ['l2']
                                   },
            "Random Forest": {'n_estimators': [100],
                              'max_depth': [None, 10, 20]
                                            
                              },
            "SVM": { 'C': [0.1, 1, 10]}
                                 ,
            "Gradient Boosting":{
                                    'n_estimators': [100]}
                        #                       'learning_rate': [0.01, 0.1, 0.5],
                        #                        'max_depth': [3, 5, 7]
                        #                     },
                       # "KNN": {'n_neighbors': [3, 5, 7, 9]}
                        #          'weights': ['uniform', 'distance']
                        #         }
                      }
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)
            print(model_report)
            
            ## To get best model score from dict
# Find the best model based on accuracy
            accuracy_scores = {model_name: metrics['Accuracy'] for model_name, metrics in model_report.items()}
            best_model_name = max(accuracy_scores, key=accuracy_scores.get)
            best_model_score = accuracy_scores[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            print(f"Best Model: {best_model_name}, Accuracy: {best_model_score:.4f}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

# Make final predictions
            predicted = best_model.predict(X_test)
            acc = accuracy_score(y_test, predicted)
            print(f"Final Model Accuracy: {acc:.4f}")

            return acc
         

        except Exception as e:
            
            raise CustomException(e, sys)
        
   


    


  
   









           
            

        























# random_state = 42

# Define your models and parameters
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Random Forest": RandomForestClassifier(class_weight={0: 1, 1: 100}),
#     "SVM": SVC(class_weight={0: 1, 1: 100}),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
#     "KNN": KNeighborsClassifier()
# }

# params = {
#     "Logistic Regression": {
#         'C': [0.001, 0.01, 0.1, 1, 10],
#         'penalty': ['l1', 'l2'],
#         'solver': ['liblinear']
#     },
#     "Random Forest": {
#         'n_estimators': [10, 50, 100],
#         'max_depth': [None, 10, 20],
#         'class_weight': [{0: 1, 1: 100}]
#     },
#     "SVM": {
#         'C': [0.1, 1, 10],
#         'kernel': ['linear', 'rbf'],
#         'class_weight': [{0: 1, 1: 100}]
#     },
#     "Gradient Boosting": {
#         'n_estimators': [50, 100],
#         'learning_rate': [0.01, 0.1, 0.5],
#         'max_depth': [3, 5, 7]
#     },
#     "KNN": {
#         'n_neighbors': [3, 5, 7, 9],
#         'weights': ['uniform', 'distance']
#     }
# }

# def evaluate_models(X_train, y_train, X_test, y_test, models, params):
#     report = []
#     best_model = None
#     best_score = 0  # Initialize to track the best score

#     for model_name, model in models.items():
#         para = params.get(model_name, {})

#         # Perform GridSearchCV to find the best parameters
#         gs = GridSearchCV(model, para, cv=3, n_jobs=-1, scoring='accuracy')  # You can change scoring as needed
#         gs.fit(X_train, y_train)

#         # Set the best parameters found
#         model.set_params(**gs.best_params_)
#         model.fit(X_train, y_train)  # Train model with best parameters

#         # Save the model to disk as a pickle file
#         # with open(f'{model_name.replace(" ", "_")}.pkl', 'wb') as file:
#         #     pickle.dump(model, file)

#         save_dir = 'D:\\Git_Hub_Project_Final'
#         os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn'
#         # Clean the model name for use in the filename
#         model_name_cleaned = model_name.replace(" ", "_")
#         save_path = os.path.join(save_dir, f'{model_name_cleaned}.pkl')

# # Save the model to disk




# #

#         # Make predictions
#         y_test_pred = model.predict(X_test)

#         # Calculate accuracy score
#         acc = accuracy_score(y_test, y_test_pred)

#         # Generate classification report as a dictionary
#         report_dict = classification_report(y_test, y_test_pred, output_dict=True)

#         # Store results in the report
#         report.append({
#             'Model': model_name,
#             'Best Parameters': gs.best_params_,
#             'Accuracy': acc,
#             'Precision': report_dict['weighted avg']['precision'],
#             'Recall': report_dict['weighted avg']['recall'],
#             'F1-Score': report_dict['weighted avg']['f1-score'],
#             'Support': report_dict['weighted avg']['support'],
#             'False Positives': sum((y_test == 0) & (y_test_pred == 1)),
#             'False Negatives': sum((y_test == 1) & (y_test_pred == 0))
#         })

#         # Check if this model is the best
#         if acc > best_score:
#             best_score = acc
#             best_model = model

#     # Convert report to DataFrame
#     report_df = pd.DataFrame(report)

#     return report_df, best_model, gs.best_params_

# # Evaluate the models
# model_report_df, best_model, best_params = evaluate_models(X_train, y_train, X_test, y_test, models, params)

# # Print the results
# print("Model Report:")
# print(model_report_df)

# print("\nBest Model:")
# print(best_model)

# print("\nBest Parameters:")
# print(best_params)
