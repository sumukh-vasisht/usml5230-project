''' IMPORTS '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import random
import yaml as y
import pickle

import warnings
warnings.filterwarnings('ignore')

''' SPOTIPY TOKENS '''
client_id = '705dbe6ad8834b4fbee893b05bf70e11'
client_secret = '64d5ed61ae124a9cafc9fa7aa07aca89'

''' SPOTIPY API '''
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id = client_id,
                                                           client_secret = client_secret))

''' READ DATA '''
data = pd.read_csv('data/data.csv')
genre_data = pd.read_csv('data/data_by_genres.csv')
year_data = pd.read_csv('data/data_by_year.csv')
artist_data = pd.read_csv('data/data_by_artist.csv')

recommended_tracks = []
track_list = []
init_page = True
track_index = 0

app = Flask(__name__)

''' Columns of data to be considered '''
columns_of_interest = ['danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo', 'duration_ms', 'popularity', 'year']

''' Clustering pipeline '''
# track_clustering_pipeline = Pipeline([('scaler', StandardScaler()),
#                                        ('kmeans', KMeans(n_clusters = 20))],
#                                      verbose = False)
# X = data[columns_of_interest]
# track_clustering_pipeline.fit(X)

# with open('kmeans.pkl', 'wb') as f:
#     pickle.dump(track_clustering_pipeline, f)

with open('kmeans.pkl', 'rb') as f:
    track_clustering_pipeline = pickle.load(f)

'''
Funtion to fetch song details from Spotipy API

Input Params: song_name str, year str
Returns: metadata of song
'''
def get_track_details_from_spotify(song_name, year):

  track_data = {}

  query = 'track: ' + song_name + ' year: ' + str(year)
  results = sp.search(query, limit=1)
  # If search result empty, return None
  if results['tracks']['items'] == []:
    return None

  # Get results
  results = results['tracks']['items'][0]
  # Get track id of the song
  track_id = results['id']
  # Get audio features of the song
  audio_features = sp.audio_features(track_id)[0]

  # Fill track_data dict
  track_data['name'] = [song_name]
  track_data['year'] = [year]
  track_data['explicit'] = [int(results['explicit'])]
  track_data['duration_ms'] = [results['duration_ms']]
  track_data['popularity'] = [results['popularity']]

  # Fill audio features
  for key, value in audio_features.items():
    track_data[key] = value

  # Resurt track_data dict
  return pd.DataFrame(track_data)

'''
Get all details of track
First search local database for the track. This reduces API calls.
Input params: 
track dict, spotify_data pd DF
Returns:
track details
'''
def get_track_data(track, spotify_data):
    
    # Search local database
    try:
        track_data = spotify_data[(spotify_data['name'] == track['name']) 
                                & (spotify_data['year'] == track['year'])].iloc[0]
        return track_data
    
    # If track not present in local database, call spotipy api
    except:
        return get_track_details_from_spotify(track['name'], track['year'])
        

'''
Get mean vector of a track
Input params: 
track_list list, spotify_data pd DF
Returns:
mean vector of tracks
'''
def get_mean_vec(track_list, spotify_data):
    
    print(track_list)

    track_vectors = []
    
    for track in track_list:
        track_data = get_track_data(track, spotify_data)
        print(track_data)
        if track_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(track['name']))
            continue
        track_vector = track_data[columns_of_interest].values
        # print('---------------------------')
        # print(track_vector)
        # print('---------------------------')
        track_vectors.append(track_vector) 
    # print(track_vectors) 
    # print('---------------------------')
    
    track_matrix = np.array(list(track_vectors))
    # print(track_matrix)
    # print('---------------------------')
    # for i in track_matrix[0][0]:
    #     print(type(i))
    # print(np.mean(track_matrix, axis=0))
    return np.mean(track_matrix, axis=0)

'''
Recommend songs using already clustered data
Input params: 
track_list list, spotify_data pd DF, number of tracks to recommend int
Returns:
recommended tracks
'''
def recommend_tracks(track_list, spotify_data, n_tracks=50):
    
    columns = ['name', 'year', 'artists']
    track_dict = {
        'name': [],
        'year': []
    }
    for dictionary in track_list:
      track_dict['name'].append(dictionary['name'])
      track_dict['year'].append(dictionary['year'])
    # print(track_dict)

    # Get track center, scale data and get recommendations    
    track_center = get_mean_vec(track_list, spotify_data)
    scaler = track_clustering_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[columns_of_interest])
    scaled_track_center = scaler.transform(track_center.reshape(1, -1))
    distances = cdist(scaled_track_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_tracks][0])
    
    # Get recommended tracks
    # print(type(index[0]))
    # print(index)
    recommended_tracks = spotify_data.iloc[index]
    # print(recommended_tracks)
    # Remove all those tracks already in the playlist
    recommended_tracks = recommended_tracks[~recommended_tracks['name'].isin(track_dict['name'])]
    # Return recommended tracks
    return recommended_tracks[columns].to_dict(orient='records')

@app.route('/', methods=['GET','POST'])
def index():
    return render_template("index.html")

@app.route('/clustering', methods=['GET','POST'])
def clustering():
    return render_template("clustering.html")

@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    top_songs = pd.read_csv('data/top_songs.csv')
    top_genres = pd.read_csv('data/top_genres.csv')
    top_artists = pd.read_csv('data/top_artists.csv')

    songs_list = top_songs['name'].tolist()
    songs_popularity = top_songs['popularity'].tolist()

    genres_list = top_genres['genres'].tolist()
    genres_popularity = top_genres['popularity'].tolist()

    artists_list = top_artists['artists'].tolist()
    artists_popularity = top_artists['popularity'].tolist()

    return render_template("dashboard.html", sl = songs_list, sp = songs_popularity, 
    gl = genres_list, gp = genres_popularity,
    al = artists_list, ap = artists_popularity)

@app.route('/build_playlist', methods=['GET','POST'])
def build_playlist():

    global track_list 
    global recommended_tracks
    global init_page

    if request.method == "POST":

        track_name = request.form['song_name'].lower()
        # year = int(request.form['year'])
        year = 2021
        
        track_list.append({'name': track_name, 'year': year}) 
        recommended_tracks = recommend_tracks(track_list, data)

        tracks_added = []
        for track in track_list:
            tracks_added.append(track['name'])
        print("TRACKS ADDED")
        print(tracks_added)

        print('Number of recommended tracks: ', len(recommended_tracks))
        items_to_delete = []
        for i in range(len(recommended_tracks)):
            print(i)
            if(recommended_tracks[i]['name'].lower() in tracks_added):
                items_to_delete.append(i)
        for item in items_to_delete:
            del recommended_tracks[item]
        
        print('----------------------------------')
        print(recommended_tracks)
        print(len(recommended_tracks))

        return render_template("build_playlist_2.html", track_list = track_list, 
        recommended_tracks = recommended_tracks,
        len_track_list = len(track_list),
        num_recs = len(recommended_tracks))

    return render_template("build_playlist.html")

@app.route('/build_playlist_2', methods=['GET', 'POST'])
def build_playlist_2():

    if request.method == "POST":

        global track_list
        global recommended_tracks
        global track_index

        # index = request.form['get_index']
        new_track_index = request.form['hsname']
        print('NEW TRACK INDEX')
        print(new_track_index)
        new_added_track = recommended_tracks[int(new_track_index)]
        print('NEW ADDED TRACK')
        print(new_added_track)
        track_list.append({'name': new_added_track['name'].lower(), 'year': new_added_track['year']})
            
        print('NEW TRACK LIST')
        print(track_list)

        recommended_tracks = recommend_tracks(track_list, data)
        print('RECOMMENDED TRACKS')
        print(recommended_tracks)

        tracks_added = []
        for track in track_list:
                tracks_added.append(track['name'])
        print("TRACKS ADDED")
        print(tracks_added)

        print('Number of recommended tracks: ', len(recommended_tracks))
        items_to_delete = []
        for i in range(len(recommended_tracks)):
            if(recommended_tracks[i]['name'].lower() in tracks_added):
                items_to_delete.append(i)
        print('ITEMS TO DELETE')
        print(items_to_delete)
        items_to_delete.sort(reverse = True)
        for item in items_to_delete:
            print(item)
            del recommended_tracks[item]
            
        return render_template("build_playlist_2.html", track_list = track_list, 
        recommended_tracks = recommended_tracks,
        len_track_list = len(track_list),
        num_recs = len(recommended_tracks))

# @app.route('/reload', methods=['GET', 'POST'])
# def reload():
#     if request.method == "POST":

#         print("ENTERED RELOAD")

#         global index

#         index = int(str(request.get_json()))
#         print(index)

#         return render_template('loading_page.html')

if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug = True)