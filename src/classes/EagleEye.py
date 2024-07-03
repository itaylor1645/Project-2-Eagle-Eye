import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

    def clean_data(self):
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates

    def get_data(self):
        return self.data
    
    def split_data(self, target='y'):
        print("Splitting data...")
        self.X = self.data.drop(target, axis=1)
        self.y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)

    def train_randomforest_model(self):
        # Check if the model.pkl file exists if it does then load the model, if it doesnt then train the model
        print("Training model...")
        # if os.path.exists('model.pkl'):
        #     self.model = joblib.load('model.pkl')
        # else:
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
        #joblib.dump(self.model, 'model.pkl')

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
    
    def get_accuracy_metrics(self, y, y_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        #classification_report = classification_report( y, y_pred )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
#            "classification_report": classification_report, 
        }

    def evaluate_model(self):
        print("Evaluating model...")
        print("Making predictions...")
        self.y_pred_test = self.predict(self.X_test)
        self.y_pred_train = self.predict(self.X_train)
        print("Getting accuracy metrics...")

        evaluation = { "Test Data Scores": self.get_accuracy_metrics(self.y_test, self.y_pred_test), "Train Data Scores": self.get_accuracy_metrics(self.y_train, self.y_pred_train)}
        print(f"--- Test Data Scores ---\n {evaluation["Test Data Scores"]}")
        print(f"--- Train Data Scores ---\n {evaluation["Train Data Scores"]}")

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
        
