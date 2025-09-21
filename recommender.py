import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

@st.cache_data
def read_process_data():
    # Data loading and preprocessing
    trips = pd.read_csv('https://sds-aau.github.io/SDS-master/M1/data/trips.csv')

    trips['date_end'] = pd.to_datetime(trips.date_end, errors='coerce')
    trips['date_start'] = pd.to_datetime(trips.date_start, errors='coerce')
    first = trips['date_start'].quantile(0.05)
    last = trips['date_end'].quantile(0.95)
    trips = trips[(trips.date_start >= first) & (trips.date_end <= last)]

    # Encode user and place IDs
    le_user = LabelEncoder()
    le_place = LabelEncoder()
    trips['username_id'] = le_user.fit_transform(trips['username'])
    trips['place_slug_id'] = le_place.fit_transform(trips['place_slug'])

    # Construct the sparse matrix
    ones = np.ones(len(trips), np.uint32)
    matrix = ss.coo_matrix((ones, (trips['username_id'], trips['place_slug_id'])))

    # Decompose the matrix with SVD
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    matrix_users = svd.fit_transform(matrix)
    matrix_places = svd.fit_transform(matrix.T)

    # Calculate cosine distance for places
    cosine_distance_matrix_places = cosine_distances(matrix_places)

    return trips, le_user, le_place, matrix, svd, matrix_users, matrix_places, cosine_distance_matrix_places

trips, le_user, le_place, matrix, svd, matrix_users, matrix_places, cosine_distance_matrix_places = read_process_data()

def similar_place(place, n):
    """
    Function to find similar places to the selected one.
    place: name of the place (str)
    n: number of similar places to return
    """
    ix = le_place.transform([place])[0]
    sim_places = le_place.inverse_transform(np.argsort(cosine_distance_matrix_places[ix,:])[:n+1])
    return sim_places[1:]

st.title('Nomad Place Recommender')

one_city = st.selectbox('Select Place', trips.place_slug.unique())
n_recs_c = st.slider('How many recs?', 1, 20, 2)

if st.button('Recommend Cities - click!'):
    recommended_places = similar_place(one_city, n_recs_c)
    st.write(recommended_places)

def similar_user_place(username, n):
    """
    Function to recommend places for a specific user.
    username: name of the user (str)
    n: number of recommended places
    """
    u_id = le_user.transform([username])[0]
    user_places_ids = trips[trips.username_id == u_id]['place_slug_id'].unique()
    user_vector_trips = np.mean(matrix_places[user_places_ids], axis=0)
    closest_for_user = cosine_distances(user_vector_trips.reshape(1,5), matrix_places)
    sim_places = le_place.inverse_transform(np.argsort(closest_for_user[0])[:n])
    return sim_places

one_user = st.selectbox('Select User', trips.username.unique())
if one_user:
    st.write(trips[trips.username == one_user]['place_slug'].unique())

n_recs_u = st.slider('How many recs? for user', 1, 20, 2)

if st.button('Recommend for a user - click!'):
    recommended_cities = similar_user_place(one_user, n_recs_u)
    st.write(recommended_cities)

    # Displaying a map of the recommended cities
    trips_viz = trips[trips.place_slug.isin(recommended_cities)]
    trips_viz = trips_viz.drop_duplicates(subset=['place_slug'])  # Keep unique places
    st.map(trips_viz[['latitude', 'longitude']])  # Make sure your data contains latitude and longitude columns
