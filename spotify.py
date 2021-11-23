import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

load_dotenv()
cid = os.getenv('SPOTIFY_CLIENT_ID')
secret = os.getenv('SPOTIFY_CLIENT_SECRET')

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

def get_playlist(creator, playlist_id):
    playlist_features_list = ["artist","album","track_name","track_id","danceability","energy","key","loudness","mode","speechiness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature"]
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        playlist_features = {}
        
        # Metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]
        
        # Audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[4:]:
            playlist_features[feature] = audio_features[feature]
        
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)

    return playlist_df

beatles = get_playlist("andream4273","1Gf0v4DneJjq3adPSiNVe6")
beatles["track_name"] = beatles["track_name"].str.replace(r'-[^-]+$', "", regex=True)

charting = get_playlist("Spotify", "37i9dQZF1DX5Ejj0EkURtP")
