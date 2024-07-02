from classes.SpotifyClient import SpotifyClient
from dotenv import load_dotenv
import os
from flask import Flask, redirect, request, session, url_for, render_template
from classes.EagleEye import EagleEye


load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_mapping(
    SECRET_KEY='dev',
    TEMPLATES_AUTO_RELOAD=True,
)

app.secret_key = os.urandom(24)  # For session management

# Initialize Spotify client

#TODO: We will use the SpotifyClient class to interact with the Spotify API to get the latest metrics of a song to have more up to date information.
# We will only use this if a valid API key is provided in the .env file, otherwise we will train on the existing data we have in the csv file. 
spotify = SpotifyClient()

def train_model():
    EagleEye = EagleEye("../Resources/uncleaned_data.csv")
    EagleEye.train_model()
    return EagleEye

@app.route('/')
def index():
    # Check if spotify api key is in the .env file
    if os.environ.get('SPOTIFY_CLIENT_ID') is None or os.environ.get('SPOTIFY_CLIENT_SECRET') is None:
        return redirect('/home')
    
    # If spotify API token is configured redirect user to Spotify authorization URL
    return redirect(spotify.OAuth.get_authorize_url())

@app.route('/callback')
def callback():
    
    # Get the authorization code from the callback URL
    code = request.args.get('code')

    # Get the access token using the authorization code and set it in the session
    session['spotify_token_info'] = spotify.OAuth.get_access_token(code)
    return redirect(url_for('home'))

@app.route('/profile')
def profile():
    
    # Get the access token from the session
    spotify_token_info = session.get('spotify_token_info', None)
    if not spotify_token_info:
        return redirect('/')
    
    # Set the access token in the Spotify client
    spotify.set_access_token(spotify_token_info) 
    
    # Get user profile and display it
    user_profile = spotify.get_user()
    return f"User profile: {user_profile}"

@app.route('/submit-song', methods=['POST'])
def submit_song():
    file = request.files['song']
    if file:
        # Save the uploaded file
        filepath = './uploaded_song'
        file.save(filepath)

        # Analyze the audio file
        eagleEye = EagleEye()
        metrics = eagleEye.analyze_audio(filepath)

    return render_template('home.html', data=metrics)

@app.route('/home')
def home():
    #EagleEye = train_model()

    # TODO: Create an input and button to upload a new song to predict the popularity of the song
    # TODO: Process the uploaded song and gather the X features of the song to pass into the predict function
    #X_data = EagleEye.analyze_audio(filepath)
    #EagleEye.predict(X_data)
    # TODO: Create a table to display the predictions of the songs and the information about the songs (tempo, danceability, etc.)

    data = "<p>Some placeholder text</p>"
    return render_template('home.html', data=data)

if __name__ == '__main__':
  app.run(debug=True, port=os.environ.get('PORT', 5000))
