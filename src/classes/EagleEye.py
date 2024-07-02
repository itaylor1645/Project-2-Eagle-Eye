import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
        self.data['y'] = self.data['y'].astype(float)
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

    def train_model(self):
        # Check if the model.pkl file exists if it does then load the model, if it doesnt then train the model
        print("Training model...")
        if os.path.exists('model.pkl'):
            self.model = joblib.load('model.pkl')
        else:
          self.model = RandomForestRegressor()
          self.model.fit(self.X_train, self.y_train)
          joblib.dump(self.model, 'model.pkl')

    def predict(self, X_data=None):
        if not X_data:
            X_data = self.X_test
        return self.model.predict(X_data)

    def evaluate_model(self):
        print("Evaluating model...")
        print("Making predictions...")
        self.y_pred = self.predict()
        print("Calculating mean_squared_error...")
        self.mean_squared_error = mean_squared_error(self.y_test, self.y_pred)
        print("Calculating mean_absolute_error...")
        self.mean_absolute_error = mean_absolute_error(self.y_test, self.y_pred)
        print("Calculating r2_score...")
        self.r2_score = r2_score(self.y_test, self.y_pred)

        return {
            "mean_squared_error": self.mean_squared_error,
            "mean_absolute_error": self.mean_absolute_error,
            "r2_score": self.r2_score,
        }

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
        
