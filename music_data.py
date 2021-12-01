import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import requests

from bs4 import BeautifulSoup

load_dotenv()
spotify_cid = os.getenv('SPOTIFY_CLIENT_ID')
spotify_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
musixmatch_auth = os.getenv('MUSIXMATCH_AUTH_CODE')

client_credentials_manager = SpotifyClientCredentials(client_id=spotify_cid, client_secret=spotify_secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# Use Spotify API to get all the songs from a given playlist
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

# Get the Beatles and top charting songs
beatles = get_playlist("andream4273","1Gf0v4DneJjq3adPSiNVe6")
beatles["track_name"] = beatles["track_name"].str.replace(r'-[^-]+$', "", regex=True)
vivien = get_playlist("Vivien", "4a6u0ZVG0FWYAJHVggnHAh")
william = get_playlist("William", "1ukuCLLRLSXE7WYWlbEq2n")

# Save playlists and info to CSVs
beatles.to_csv("data/beatles.csv")
vivien.to_csv("data/vivien.csv")
william.to_csv("data/william.csv")

# Get the lyrics for a given song
def get_lyrics(track, artist):
    track = track.replace(" ", "%20")
    artist = artist.replace(" ", "%20")
    link = "https://api.musixmatch.com/ws/1.1/matcher.lyrics.get?format=json&callback=callback&q_track="+track+"&q_artist="+artist+"&apikey="+musixmatch_auth
    response = requests.get(link)
    try:
        return response.json()['message']['body']['lyrics']['lyrics_body'].split("...")[0]
    except:
        return None

# Beautiful soup webscraping, doesn't work tho
# def get_lyrics(artistname, songname):
#     artistname2 = str(artistname.replace(' ','-')) if ' ' in artistname else str(artistname)
#     songname2 = str(songname.replace(' ','-')) if ' ' in songname else str(songname)
#     page = requests.get('https://genius.com/'+ artistname2 + '-' + songname2 + '-' + 'lyrics')
#     html = BeautifulSoup(page.text, 'html.parser')
#     lyrics1 = html.find("div", class_="lyrics")
#     lyrics2 = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-2 jgQsqn")
#     if lyrics1:
#         lyrics = lyrics1.get_text()
#     elif lyrics2:
#         lyrics = lyrics2.get_text()
#     elif lyrics1 == lyrics2 == None:
#         lyrics = None
#     return lyrics

# Get the corresponding lyrics to the Beatles and top charting songs
#beatles['lyrics'] = beatles.apply(lambda track: get_lyrics(track['track_name'], track['artist']), axis=1)
#charting['lyrics'] = charting.apply(lambda track: get_lyrics(track['track_name'], track['artist']), axis=1)