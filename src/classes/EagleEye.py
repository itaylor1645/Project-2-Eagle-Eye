import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import librosa
import numpy as np
import joblib
import os
from pydub import AudioSegment


class EagleEye:
    def __init__(self, csv_file=None, target='y'):
        if csv_file:
          self.load_csv(csv_file, target)

    def convert_audio(self, file_path):
        # Convert the audio file to WAV format if necessary
        audio = AudioSegment.from_file(file_path)
        wav_file_path = file_path + '.wav'
        audio.export(wav_file_path, format='wav')
        return wav_file_path
    
    # Load audio file using librosa
    def analyze_audio(self, file_path):

      wav_file_path = self.convert_audio(file_path)

      y, sr = librosa.load(wav_file_path)

      # Calculate duration in milliseconds
      duration_ms = librosa.get_duration(y=y, sr=sr) * 1000

      # Calculate danceability and other features using Spotify API or predefined methods
      # Note: Implementing danceability, valence, and other Spotify-specific metrics is non-trivial without their API
      tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
      loudness = np.mean(librosa.feature.rms(y=y))  # Approximation
      key = librosa.feature.chroma_stft(y=y, sr=sr)
      key = np.argmax(np.mean(key, axis=1))
      mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
      speechiness = np.mean(mfccs)  # Approximation
      chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
      spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
      zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
      rms = librosa.feature.rms(y=y)

      danceability = np.mean(zero_crossing_rate) * np.mean(rms) * tempo
      energy = np.mean(rms) * tempo

      # Acousticness, instrumentalness, liveness, valence are typically Spotify-specific
      # so, let's use dummy values here and later replace with actual API calls
      acousticness = 0.5
      instrumentalness = 0.5
      liveness = 0.5
      valence = 0.5
      time_signature = 4  # Default to 4/4 time

      # Clean up temporary files
      os.remove(file_path)
      os.remove(wav_file_path)

      return {
          "duration_ms": duration_ms,
          "tempo": tempo,
          "key": key,
          "loudness": loudness,
          "speechiness": speechiness,
          "acousticness": acousticness,
          "instrumentalness": instrumentalness,
          "liveness": liveness,
          "valence": valence,
          "time_signature": time_signature,
          "danceability": danceability,
          "energy": energy
      }


    def load_csv(self, csv_file, target='y'):
        self.data = pd.read_csv(csv_file)
        self.data.rename(columns={target: 'y'}, inplace=True)
        self.data['y'] = self.label_encode(self.data['y'])
        self.split_data()
        self.remove_outliers_by_zscore(self.X_train.columns.tolist())
        self.sample_data()

    def split_data(self, target='y'):
        print("Splitting data...")
        self.X = self.data.drop(target, axis=1)
        self.y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)

    def remove_outliers_by_zscore(self, features, zscore_threshold=3, enabled=True):
        if enabled:
            print(f"Removing outliers from the training set based on a zscore tolerance of {zscore_threshold}...")   
            # Identify outliers based on Z-score threshold for X_train
            z_scores = np.abs((self.X_train[features] - self.X_train[features].mean()) / self.X_train[features].std()) 
            outlier_mask_X = (z_scores > zscore_threshold).any(axis=1)
            self.X_train = self.X_train[~outlier_mask_X]
            self.y_train = self.y_train[~outlier_mask_X]
        else:
            print("Outlier removal is disabled. Skipping...")

    def sample_data(self, undersample=False):
        if undersample:
            print("Undersampling Data...")
            train_data = pd.concat([self.X_train, self.y_train], axis=1)
            min_class_size = train_data['y'].value_counts().min()
            
            if min_class_size <= 1:
                # Handle case where minimum class size is too small to sample
                print("Warning: Minimum class size is too small for undersampling.")
                return
            
            rus = RandomUnderSampler(sampling_strategy='auto', random_state=1)
            try:
                self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
            except Exception as e:
                print(f"Error during undersampling: {e}")
        else:
            print("Synthetic Minority Oversampling Technique (SMOTE)")
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def train_randomforest_model(self, perform_random_search=True, retrain_model=True):
        # Load model if it is already trained and exists
        if not retrain_model and os.path.exists('model.pkl') and not perform_random_search:
            self.model = joblib.load('model.pkl')
            print("Model loaded from model.pkl")
        else:
            self.model = RandomForestClassifier()

            # Perform RandomizedSearchCV to find the best hyperparameters
            if perform_random_search:
                param_dist = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
                # Create a RandomizedSearchCV object
                print("Finding best parameters...")
                random_search = RandomizedSearchCV(estimator=self.model, param_distributions=param_dist,
                                                n_iter=100, cv=3, n_jobs=-1, verbose=2, random_state=42)
                random_search.fit(self.X_train, self.y_train)
                self.model = random_search.best_estimator_
                best_params = random_search.best_params_
                print(f"Best Params: {best_params}")
                joblib.dump(self.model, 'model.pkl')
                print("Model saved to model.pkl")

            # Train with default parameters if RandomizedSearchCV is disabled
            else:
                # Train with default parameters
                print("Training model...")
                self.model.fit(self.X_train, self.y_train)

                # Save the model to disk as model.pkl so that we dont have to train it upon every run. 
                joblib.dump(self.model, 'model.pkl')
                print("Default model trained and saved to model.pkl")

    def train_xgboost_model(self):
        from xgboost import XGBClassifier
        self.y_train = self.label_encode(self.y_train)
        self.model = XGBClassifier()
        self.model.fit(self.X_train, self.y_train)

    def train_knn_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier()
        self.model.fit(self.X_train, self.y_train)

    def train_svm_model(self):
        from sklearn.svm import SVC
        self.model = SVC()
        self.model.fit(self.X_train, self.y_train)
    
    def train_logistic_regression_model(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_data=None):
        if X_data is None:
            X_data = self.X_test
        return self.model.predict(X_data)
    
    def label_encode(self, y):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        return le.fit_transform(y)
    
    def get_accuracy_metrics(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        class_report = classification_report(y_true, y_pred, target_names=[
            'acoustic', 'alternative', 'blues', 'children', 'classical', 'country', 'electronic', 'folk',
            'funk', 'hip-hop', 'jazz', 'latin', 'metal', 'miscellaneous', 'pop', 'punk', 'r&b', 'reggae',
            'regional', 'religious', 'rock', 'world'
        ])

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": class_report,  # Include the classification report as a string
        }

    def evaluate_model(self, calculate_feature_importance=True):
        print("Evaluating model...")
        print("Making predictions...")
        self.y_pred_test = self.model.predict(self.X_test)
        self.y_pred_train = self.model.predict(self.X_train)
        print("Getting accuracy metrics...")

        evaluation = {
            "Test Data Scores": self.get_accuracy_metrics(self.y_test, self.y_pred_test),
            "Train Data Scores": self.get_accuracy_metrics(self.y_train, self.y_pred_train)
        }

        if calculate_feature_importance:
            # Get feature importances
            if isinstance(self.model, RandomForestClassifier):
                feature_importances = self.model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': self.X_train.columns,
                    'Importance': feature_importances
                })
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
                print("--- Feature Importances ---")
                print(importance_df)
                print("---\n")
            else:
                print("Feature importance is only available for RandomForestClassifier.")

        print("--- Test Data Scores ---")
        for key, value in evaluation["Test Data Scores"].items():
            if key == "classification_report":
                print(f"{key}:\n{value}\n")
            else:
                print(f"{key}: {value}")
        print("---\n")

        print("--- Train Data Scores ---")
        for key, value in evaluation["Train Data Scores"].items():
            if key == "classification_report":
                print(f"{key}:\n{value}\n")
            else:
                print(f"{key}: {value}")
        print("---\n")

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
    
