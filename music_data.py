import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import requests
import re
import json

load_dotenv()
spotify_cid = os.getenv('SPOTIFY_CLIENT_ID')
spotify_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
musixmatch_auth = os.getenv('MUSIXMATCH_AUTH_CODE')
IBM_API_URL = os.getenv('API_URL')
IBM_API_KEY = os.getenv('API_KEY')

client_credentials_manager = SpotifyClientCredentials(client_id=spotify_cid, client_secret=spotify_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

all_genres = sp.recommendation_genre_seeds()["genres"]

with open('features.json') as f:
    features = json.load(f)


def calculate_sentiment(text):
    res = requests.post(
        IBM_API_URL,
        auth=(
            'apikey',
            IBM_API_KEY
        ),
        params={
            'return_analyzed_text': True
        },
        json={
            'text': text,
            'features': features
        }
    )
    res_json = res.json()
    emotions = res_json['emotion']['document']['emotion']
    sentiment = res_json['sentiment']['document']['score']

    return emotions['anger'], emotions['disgust'], emotions['fear'], emotions['joy'], emotions['sadness'], sentiment


# Use Spotify API to get all the songs from a given playlist
def get_playlist(playlist_id):
    playlist_features_list = ["artist", "album", "track_name", "track_id", "danceability", "energy", "key", "loudness",
                              "mode", "speechiness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
                              "time_signature"]
    playlist_df = pd.DataFrame(columns=playlist_features_list + all_genres)

    playlist = sp.user_playlist_tracks(playlist_id=playlist_id)

    while playlist:
        for track in playlist['items']:
            anger, disgust, fear, joy, sadness, sentiment = get_lyrics_sentiment(track['track']['name'],
                                                                                 track['track']['album']['artists'][0][
                                                                                     'name'])
            playlist_features = {}

            # Metadata
            playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
            playlist_features["album"] = track["track"]["album"]["name"]
            playlist_features["track_name"] = track["track"]["name"]
            playlist_features["track_id"] = track["track"]["id"]
            playlist_features["popularity"] = track["track"]["popularity"]
            playlist_features['sentiment'] = sentiment
            playlist_features['anger'] = anger
            playlist_features['disgust'] = disgust
            playlist_features['fear'] = fear
            playlist_features['joy'] = joy
            playlist_features['sadness'] = sadness

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
                track_df = pd.DataFrame(playlist_features, index=[0])
                playlist_df = pd.concat([playlist_df, track_df], ignore_index=True)

        # Pagination
        if playlist['next']:
            playlist = sp.next(playlist)
        else:
            playlist = None

    return playlist_df

# Get the lyrics for a given song
def get_lyrics_sentiment(track, artist):
    track = track.replace(" ", "%20")
    artist = artist.replace(" ", "%20")
    link = "https://api.musixmatch.com/ws/1.1/matcher.lyrics.get?format=json&callback=callback&q_track=" + track + "&q_artist=" + artist + "&apikey=" + musixmatch_auth
    response = requests.get(link)
    try:
        lyrics = response.json()['message']['body']['lyrics']['lyrics_body'].split("...")[0]
        return calculate_sentiment(lyrics)
    except:
        return 0.5, 0.5, 0.5, 0.5, 0.5, 0 # if lyrics not returned, do neutral (?) default values


# Get and save playlist data to csv
# beatles = get_playlist("andream4273", "1Gf0v4DneJjq3adPSiNVe6")
# beatles.to_csv("data/beatles.csv")

# vivian = get_playlist("Vivian", "4a6u0ZVG0FWYAJHVggnHAh")
# vivian.to_csv("data/vivian.csv")

# william = get_playlist("William", "1ukuCLLRLSXE7WYWlbEq2n")
# william.to_csv("data/william.csv")


def save_playlist(spotify_url):
    playlist_id = re.search('playlist\/(\w+)?', spotify_url).group(1)
    playlist = get_playlist(playlist_id)
    filename = "data/" + playlist_id + ".csv"
    playlist.to_csv(filename)
    return filename