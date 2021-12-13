import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import requests
import re

load_dotenv()
spotify_cid = os.getenv('SPOTIFY_CLIENT_ID')
spotify_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
musixmatch_auth = os.getenv('MUSIXMATCH_AUTH_CODE')

client_credentials_manager = SpotifyClientCredentials(client_id=spotify_cid, client_secret=spotify_secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
all_genres = sp.recommendation_genre_seeds()["genres"]

# Use Spotify API to get all the songs from a given playlist
def get_playlist(playlist_id):
    playlist_features_list = ["artist","album","track_name","track_id","danceability","energy","key","loudness","mode","speechiness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature"]
    playlist_df = pd.DataFrame(columns = playlist_features_list + all_genres)
    playlist = sp.user_playlist_tracks(playlist_id=playlist_id)
    
    while playlist:
        for track in playlist['items']:
            playlist_features = {}
            
            # Metadata
            playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
            playlist_features["album"] = track["track"]["album"]["name"]
            playlist_features["track_name"] = track["track"]["name"]
            playlist_features["track_id"] = track["track"]["id"]
            playlist_features["popularity"] = track["track"]["popularity"]
            
            # Genres
            track_genres = set(sp.artist(track["track"]["album"]["artists"][0]["uri"])["genres"])
            for genre in all_genres:
                playlist_features[genre] = int(genre in track_genres)

            # Audio features
            audio_features = sp.audio_features(playlist_features["track_id"])[0]
            
            if audio_features:
                for feature in playlist_features_list[4:]:
                    playlist_features[feature] = audio_features[feature]

                # Concat the dfs
                track_df = pd.DataFrame(playlist_features, index = [0])
                playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
        # Pagination
        if playlist['next']:
            playlist = sp.next(playlist)
        else:
            playlist = None

    return playlist_df

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

# Get and save playlist data to csv
# beatles = get_playlist("andream4273","1Gf0v4DneJjq3adPSiNVe6")
# beatles["track_name"] = beatles["track_name"].str.replace(r'-[^-]+$', "", regex=True) # simplify song titles
# beatles['lyrics'] = beatles.apply(lambda track: get_lyrics(track['track_name'], track['artist']), axis=1)
# beatles.to_csv("data/beatles.csv")

# vivian = get_playlist("Vivian", "4a6u0ZVG0FWYAJHVggnHAh")
# vivian['lyrics'] = vivian.apply(lambda track: get_lyrics(track['track_name'], track['artist']), axis=1)
# vivian.to_csv("data/vivianTEst.csv")


# william = get_playlist("William", "1ukuCLLRLSXE7WYWlbEq2n")
# william['lyrics'] = william.apply(lambda track: get_lyrics(track['track_name'], track['artist']), axis=1)
# william.to_csv("data/william.csv")

def save_playlist(spotify_url):
    playlist_id = re.search('playlist\/(\w+)?', spotify_url).group(1)
    playlist = get_playlist(playlist_id)
    # playlist["track_name"] = playlist["track_name"].str.replace(r'-[^-]+$', "", regex=True) # for beatles songs only, simplify song titles
    playlist['lyrics'] = playlist.apply(lambda track: get_lyrics(track['track_name'], track['artist']), axis=1)
    filename = "data/" + playlist_id + ".csv"
    playlist.to_csv(filename)
    return filename