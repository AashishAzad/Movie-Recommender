import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def convertCast(obj):
    castList = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            castList.append(i['name'])
            counter += 1
        else:
            break
    return castList


def getDirector(obj):
    directorList = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            directorList.append(i['name'])
            break
    return directorList


# def recommend(movie):
#     movieIndex = new_df[new_df['title'] == movie].index[0]
#     distances = similarity[movieIndex]
#     moviesList = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
#
#     for i in moviesList:
#         print(new_df.iloc[i[0]].title)


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


# This function will convert genres in List because they are given in different Manner.
def convert(obj):
    genreList = []
    for i in ast.literal_eval(obj):
        genreList.append(i['name'])
    return genreList


movies = pd.read_csv('tmdb_5000_movies.csv')
credit = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credit, on='movie_id')

# We are gonna selects these fields from our data and remove other unnecessary columns
# genre, movie_id, keywords, title, overview, cast, crew
movies = movies[['genres', 'movie_id', 'keywords', 'title_y', 'overview', 'cast', 'crew']]
movies.rename(columns={'title_y': 'title'}, inplace=True)

# We had 3 cells of overview columns which is null that's why we are dumping it.
movies.dropna(inplace=True)

# This helps in stemming
ps = PorterStemmer()

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convertCast)
movies['crew'] = movies['crew'].apply(getDirector)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Removing spaces to create Tags
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Now the data is ready.
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Now we will create new dataframe with selected columns
movies_df = movies[['movie_id', 'title', 'tags']]

# Now we will convert tags column into string not list
movies_df['tags'] = movies_df['tags'].apply(lambda x: " ".join(x))
movies_df['tags'] = movies_df['tags'].apply(lambda x: x.lower())

movies_df['tags'] = movies_df['tags'].apply(stem)

cv = CountVectorizer(stop_words='english', max_features=5000)
vectors = cv.fit_transform(movies_df['tags']).toarray()

# cv.getFeatureName() --> this will most occurring word

similarity = cosine_similarity(vectors)

# this will create a file name "movies.pkl" which store new_df
pickle.dump(movies_df.to_dict(), open('../moviesDict.pkl', 'wb'))
pickle.dump(similarity, open('../similarity.pkl', 'wb'))
