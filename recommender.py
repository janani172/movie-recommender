import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data preparation
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(text):
    L = []
    for i in ast.literal_eval(text): # Convert string to a Python list
        L.append(i['name']) # Extract the 'name' field from each dictionary
    return L 

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def get_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []



movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(get_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))  # Remove spaces from each element
    return L1

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]

new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())


import nltk   # Importing the Natural Language Toolkit
from nltk.stem import PorterStemmer  # Importing the Porter Stemmer


ps = PorterStemmer()

def stems(text):
    T = []
    
    for i in text.split():
        T.append(ps.stem(i))   # Apply stemming to each word and add to the list
    
    return " ".join(T)  # Join the stemmed words back into a sentence


new_df.loc[:, 'tags'] = new_df['tags'].apply(stems)


# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_title):
    if movie_title not in new_df['title'].values:
        return ["Movie not found in dataset."]
    
    index = new_df[new_df['title'] == movie_title].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    return [new_df.iloc[i[0]].title for i in movie_list]
