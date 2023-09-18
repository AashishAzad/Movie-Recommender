import streamlit as st
import pickle
import pandas as pd


def recommend(movie):
    movieIndex = movies[movies['title'] == movie].index[0]
    distances = similarity[movieIndex]
    moviesList = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommendedMovies = []
    for i in moviesList:
        recommendedMovies.append(movies.iloc[i[0]].title)

    return recommendedMovies


moviesDict = pickle.load(open('moviesDict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pd.DataFrame(moviesDict)

st.title('Movie Recommender system')

selectedMovie = st.selectbox("Select Movie", movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selectedMovie)
    for i in recommendations:
        st.write(i)
