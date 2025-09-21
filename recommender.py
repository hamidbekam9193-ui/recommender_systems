import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

@st.experimental_singleton
def read_process_data():
# data and prepro
    trips = pd.read_csv('https://sds-aau.github.io/SDS-master/M1/data/trips.csv')

    trips['date_end'] = pd.to_datetime(trips.date_end, errors='coerce')
    trips['date_start'] = pd.to_datetime(trips.date_start, errors='coerce')
    first = trips['date_start'].quantile(0.05)
    last = trips['date_end'].quantile(0.95)
    trips = trips[(trips.date_start >= first) & (trips.date_end <= last)]

    # encode ids
    le_user = LabelEncoder()
    le_place = LabelEncoder()
    trips['username_id'] = le_user.fit_transform(trips['username'])
    trips['place_slug_id'] = le_place.fit_transform(trips['place_slug'])

    # construct matrix
    ones = np.ones(len(trips), np.uint32)
    matrix = ss.coo_matrix((ones, (trips['username_id'], trips['place_slug_id'])))

    # decomposition
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    matrix_users = svd.fit_transform(matrix)
    matrix_places = svd.fit_transform(matrix.T)

    # distance-matrix
    cosine_distance_matrix_places = cosine_distances(matrix_places)

    return trips, le_user, le_place, matrix, svd, matrix_users, matrix_places, cosine_distance_matrix_places

trips, le_user, le_place, matrix, svd, matrix_users, matrix_places, cosine_distance_matrix_places = read_process_data()

def similar_place(place, n):
  """
  this function performs city similarity search
  place: name of place (str)
  n: number of similar cities to print
  """
  ix = le_place.transform([place])[0]
  sim_places = le_place.inverse_transform(np.argsort(cosine_distance_matrix_places[ix,:])[:n+1])
  return sim_places[1:]

st.title('Nomad Place Recommender')

one_city = st.selectbox('Select Place', trips.place_slug.unique())
n_recs_c = st.slider('How many recs?', 1, 20, 2)

if st.button('Recommend Cities - click!'):
    st.write(similar_place(one_city, n_recs_c))


def similar_user_place(username, n):
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
    similar_cities = similar_user_place(one_user, n_recs_u)
    st.write(similar_cities)

    # adding a simple map-viz
    trips_viz = trips[trips.place_slug.isin(similar_cities)]
    trips_viz.drop_duplicates(subset=['place_slug'], inplace=True) #keep only individual place_slug observations for mapping
    st.map(trips_viz)