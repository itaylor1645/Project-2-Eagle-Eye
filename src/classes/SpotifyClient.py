import os
import spotipy

class SpotifyClient:
    def __init__(self):
        client_id = os.environ.get('SPOTIFY_CLIENT_ID')
        client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
        redirect_uri = os.environ.get('SPOTIFY_REDIRECT_URI')
        self.OAuth = spotipy.oauth2.SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope='user-library-read user-top-read')
        self.Client = spotipy.Spotify(auth_manager=self.OAuth)

    def set_access_token(self, spotify_token_info):
        self.Client = spotipy.Spotify(auth=spotify_token_info['access_token'])

    # TODO: Implement the following methods to be used through out the program

    # def get_user(self):
    #     return self.Client.current_user()
    
    # def get_user_playlists(self):
    #     return self.Client.current_user_saved_tracks()

    # def get_playlists(self):
    #     return self.Client.get_playlists()

    # def get_playlist_tracks(self, playlist_id):
    #     return self.Client.get_playlist_tracks(playlist_id)

    # def get_track(self, track_id):
    #     return self.Client.get_track(track_id)

    # def get_album(self, album_id):
    #     return self.Client.get_album(album_id)

    # def get_artist(self, artist_id):
    #     return self.Client.get_artist(artist_id)

    # def get_artist_albums(self, artist_id):
    #     return self.Client.get_artist_albums(artist_id)

    # def get_artist_top_tracks(self, artist_id):
    #     return self.Client.get_artist_top_tracks(artist_id)

    # def get_artist_related_artists(self, artist_id):
    #     return self.Client.get_artist_related_artists(artist_id)

    # def search(self, query):
    #     return self.Client.search(query)

    # def search_tracks(self, query):
    #     return self.Client.search_tracks(query)

    # def search_albums(self, query):
    #     return self.Client.search_albums(query)

