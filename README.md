# Project-2-Eagle-Eye
A tool to predict the genre of a song based on its various metrics (danceability, energy, tempo, etc.).

# Requirements:
* python 3 

# Installation Instructions
* Run the following command from the app root directory to install the required python packages.

  ``` pip install -r requirements.txt ```

# Run Appliction

Open the data_cleaning.ipynb file in jupyter notebook or vscode and run the script. This will train the model and evaluate the model scoring metrics. 

# Data Collection and Prepocessing
We found a dataset with over 114,000 songs and 114 genres from Kaggle that was retrieved from the Spotify API. This dataset included audio features such as tempo, energy, danceability, valence, and more. The dataset was very diverse in its songs containing very unique tracks we had never heard of. We dropped duplicate values of tracks in the dataset and certain columns, such as 'Unnamed: o', to prevent data leakage.

# Model Development
We implemented and compared multiple models such as Random Forest Classifiers, Support Vector Machines and K-nearest neighbors. We used automated hyperparameter tuning to determine what model would work best, which was the Random Forest Classifier. Since our dataset had many genres we decided to group them together to try and improve performance of the model. We went from 114 genres to 22 and saw a slight improvement. 

# Results
The final model's accuracy was around 43%. We were hoping to get a higher score but it proved difficult classifying multiple genres. We concluded that music is very subjective and multifaceted, making it difficult to develop a high-performing predictive model. We are still content with the results with the dataset that we used.

# Future Work
We attempted to use this model with additional songs that were not part of the dataset, however we ran into troubles with producing the same metrics that spotify provides in their data set and having the numbers match up. So we had to settle with training and testing with only the dataset that we had available to us. If given more time we would have liked to make an application that would've included the ability to search for any song.

# Resources
Dataset
https://www.kaggle.com/datasets/mohamedhamad21/spotify-tracks-dataset
Spotify API
https://developer.spotify.com/documentation/web-api
