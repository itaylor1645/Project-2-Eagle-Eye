import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import joblib


class EagleEye:
    def __init__(self, csv_file="../Resources/unclean_data.csv"):
        self.load_csv(csv_file)
    
    def load_csv(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def clean_data(self):
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates

    def get_data(self):
        return self.data
    
    def split_data(self):
        self.X = self.data.drop('target', axis=1)
        self.y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

    def train_model(self):
        # Check if the model.pkl file exists if it does then load the model, if it doesnt then train the model
        try:
            self.model = joblib.load('model.pkl')
        except:
          if not hasattr(self, 'X_train'):
              self.split_data()
          self.model = make_pipeline(StandardScaler(), RandomForestClassifier())
          self.model.fit(self.X_train, self.y_train)
          joblib.dump(self.model, 'model.pkl')

    def predict(self, X_data):
        return self.model.predict(X_data)

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.cross_val_score = cross_val_score(self.model, self.X, self.y, cv=5)
        self.classification_report = classification_report(self.y_test, self.y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.roc_auc_score = roc_auc_score(self.y_test, self.y_pred)
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred)

    def plot_roc_curve(self):
        plt.plot(self.fpr, self.tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    def get_accuracy(self):
        return self.accuracy
    
    def get_cross_val_score(self):
        return self.cross_val_score
    
    def get_classification_report(self):
        return self.classification_report
    
    def get_confusion_matrix(self):
        return self.confusion_matrix
    
    def get_roc_auc_score(self):
        return self.roc_auc_score
    
    def get_roc_curve(self):
        return self.fpr, self.tpr, self.thresholds
        
