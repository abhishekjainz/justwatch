# importing all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import opendatasets as od


import warnings; warnings.simplefilter('ignore')

#Connection to Fauna DB
import os
from dotenv import load_dotenv

load_dotenv()
FAUNA_KEY="fnAEMDlAEQACBRgdU1d3dWDNmswtpvVnDF4kERne"

from faunadb import query as q
from faunadb.client import FaunaClient
from faunadb.errors import NotFound

# Fauna Client Config
client = FaunaClient(secret=FAUNA_KEY)

# Downloading Dataset
od.download("https://www.kaggle.com/rounakbanik/the-movies-dataset")

## FIRST MODEL: POPULARITY BASED FILTERING
# This is a pandas built in function to read csv files and output the first 5 rows using "df.head()"
df = pd.read_csv("./the-movies-dataset/movies_metadata.csv")
df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# We first check how many movies have missing values for vote_count & vote_average
# the output is 6 (6 movies)
df['vote_count'].isnull().sum()
df['vote_average'].isnull().sum()
vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
votes_average = df[df['vote_average'].notnull()]['vote_average'].astype('int')
C = votes_average.mean()

# we will set m as the 85th percentile
m = vote_counts.quantile(0.85)

# We create a new column called year and split the string to only record the year of release 
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# We create a new dataframe called Qualified and it comprises of the top 85 percentile movies
# Calling 'qualified.shape' outputs (6832, 6) which means there are 6832 movies in this dataframe
qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & 
               (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')

# We use this function to calculate the weighted average of each movie
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

# We create a new column called "weighted rat" which comprises of the weighted rating 
# We sort it in a descending order and this dataframe becomes our dataset for the most 'popular movies'
qualified['weighted_rat'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('weighted_rat', ascending=False).head(250)

# This code outputs the top 15 movies of all time, based on the weighted rating approach
# print(qualified.head(15)) 

s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = df.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified

## SECOND MODEL: CONTENT BASED FILTERING
links_small = pd.read_csv("./the-movies-dataset/links_small.csv")
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
df = df.drop([19730, 29503, 35587])
df['id'] = df['id'].astype('int')
df2 = df[df['id'].isin(links_small)]
df2.shape

# we will be replacing null values with empty strings for consistency in datatypes
df2['tagline'] = df2['tagline'].fillna('') 
df2['description'] = df2['overview'] + df2['tagline']
df2['description'] = df2['description'].fillna('')

# TfidfVectorizer is a sklearn python library which transforms texts into meaningful respresentation of numbers which is used to fit
# machine learning algorithms
# we will use this library to analyze our description column 
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df2['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# setting the title of the movie as the index of the dataframe
df2.reset_index()
indices = pd.Series(df2.index, index = df2['title'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices]

credits = pd.read_csv("./the-movies-dataset/credits.csv")
keywords = pd.read_csv("./the-movies-dataset/keywords.csv")
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')

# we would now merge our current dataset with the crew and keyword datasets
df = df.merge(credits, on='id')
df = df.merge(keywords, on='id')
df['id'] = df['id'].astype('int')

df2 = df[df['id'].isin(links_small)]
df2.shape

df2['cast'] = df2['cast'].apply(literal_eval)
df2['crew'] = df2['crew'].apply(literal_eval)
df2['cast_size'] = df2['cast'].apply(lambda x: len(x))
df2['crew_size'] = df2['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
    
df2['director'] = df2['crew'].apply(get_director)

df2['cast'] = df2['cast'].apply(lambda x: [i['name'] for i in x] if isinstance (x, list) else [])
df2['cast'] = df2['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
df2['keywords'] = df2['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance (x, list) else [])

# this code basically converts the text into lower case and removes the space between the first and last name
# rationale behind this is so the machine does not get confused between 2 people having the same first name
df2['cast'] = df2['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
df2['director'] = df2['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))

# this code allows us to mention director 3 times to give it more weight relative to the entire cast
df2['director'] = df2['director'].apply(lambda x: [x,x, x])

s = df.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'

s = s.value_counts()
s[:5]

s = s[s > 1]

stemmer = SnowballStemmer('english')
stemmer.stem('cats')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

df2['keywords'] = df2['keywords'].apply(filter_keywords)
df2['keywords'] = df2['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
# this last line of code converts all keywords into lower case and removes any spaces
df2['keywords'] = df2['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# the following code joins all columns into a meta-column (from which we can dervice qualitative insights)
df2['text_analysis'] = df2['keywords'] + df2['cast'] + df2['director'] + df2['genres']
df2['text_analysis'] = df2['text_analysis'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(df2['text_analysis'])

# this is different from the earlier output as now our cosine similarity scores have changed
cosine_sim = cosine_similarity(count_matrix, count_matrix)
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

# Time to do an experiment to test the results
get_recommendations('The Dark Knight').head(10)

def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = df2.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified

# Time to do the same experiment with the same movie
#print(improved_recommendations('The Dark Knight'))

## THIRD MODEL: COLLABORATICE FILTERING
reader = Reader()
ratings = pd.read_csv("./the-movies-dataset/ratings_small.csv")
ratings.head()

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
algo.fit(trainset)
# Here we pick the first user and see what rating it has given!
ratings[ratings['userId'] == 1]
algo.predict(1, 302, 3)

## FINAL MODEL: HYBRID FILTERING
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
        
id_map = pd.read_csv("./the-movies-dataset/links_small.csv")[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(df2[['title', 'id', 'genres']], on='id').set_index('title')
id_map.head()

indices_map = id_map.set_index('id')
indices_map.head()

moods_dict_1: {"Loney": "Family", "Depressed": "Comedy", "Cheerful": "Animation",
               "Excited": "Science Fiction", "Stressed": "Romance"}
moods_dict_2: {"Loney": "Thriller", "Depressed": "Animated", "Cheerful": "Thriller",
               "Excited": "Romance", "Stressed": "Family"}

list_of_romance = ["Dilwale Dulhania Le Jayenge", "Paperman", "Sing Street", "The Handmaiden",
                   "The Way He Looks", "In a Heartbeat", "Titanic", "Silver Linings Playbook", "La La Land",
                   "Maleficent", "Her", "The Great Gatsby", "The Fault in Our Stars", "Eternal Sunshine of the Spotless Mind"]
                   
                   
list_of_family = ["Spirited Away", "Paperman", "Piper", "Wolf Children", "Feast", "Song of the Sea", "Harry Potter and the Philosopher's Stone",
                 "Up", "Inside Out", "Despicable Me", "WALL-E", "Finding Nemo", "Big Hero 6", "Monsters, Inc.",
                  "Harry Potter and the Deathly Hallows: Part 2"]

list_of_comedy = ["Dilwale Dulhania Le Jayenge", "The Intouchables", "The Grand Budapest Hotel",
                 "The Apartment", "Feast", "Deadpool", "Up", "The Wolf of Wall Street", 
                 "Inside Out", "The Hangover", "Big Hero 6", "Monsters, Inc.", "Kingsman: The Secret Service",
                 "Zootopia"]

list_of_animation = ["Spirited Away", "Howl's Moving Castle", "Princess Mononoke",
                    "Paperman", "Piper", "Wolf Children", "Song of the Sea",
                    "Presto", "Up", "Inside Out", "Despicable Me", "WALL·E "]

list_of_thriller = ["Inception", "The Dark Knight", "Se7en", "The Imitation Game", "The Prestige",
                    "Momento", "The Usual Suspects","Room", "Psycho", "Oldboy", "The Handmaiden", "The Invisible Guest",
                   "Mad Max: Fury Road", "The Dark Knight Rises", "Titanic"]

list_of_science_fiction = ["Inception", "Interstellar", "Avatar", "The Avengers", "Guardians of the Galaxy", 
                           "Mad Max: Fury Road", "The Matrix", "Iron Man", "Star Wars: The Force Awakens", 
                           "Captain America: Civil War", "The Martian", "Avengers: Age of Ultron", "The Hunger Games: Catching Fire",
                          "Logan", "X-Men: Days of Future Past"]


def recommend_movie(mood):

    if mood == "Lonely":
        try:
            return genre_recommendation('family') or genre_recommendation('thriller')
        except: 
            title = random.choice(list_of_family) or random.choice(list_of_thriller)
            return title

    elif mood == "Depressed":
        try: 
            return genre_recommendation('comedy') or genre_recommendation('animation')
        except: 
            title = random.choice(list_of_comedy) or random.choice(list_of_animation)
            return title

    elif mood == "Cheerful":
        try:
            return genre_recommendation('animation') or genre_recommendation('thriller')
        except: 
            title = random.choice(list_of_animation) or random.choice(list_of_thriller)
            return title

    elif mood == "Excited":
        try:
            return genre_recommendation('romance') or genre_recommendation('Fantasy')
        except: 
            title = random.choice(list_of_science_fiction) or random.choice(list_of_romance)
            return title

    elif mood == "Stressed":
        try:
            return genre_recommendation('romance') or genre_recommendation('family')
        except: 
            title = random.choice(list_of_romance) or random.choice(list_of_family)
            return title

def recommend_movie_backend(movie):
    title = movie
    return process(title)

def process(title):        
    userId = random.randrange(0, 100)
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
        
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
        
    movies = df2.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies.drop(movies[movies.year < '1994'].index, inplace=True)
    movies['est'] = movies['id'].apply(lambda x: algo.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    #print(movies.head(1)['title'])
    return movies.head(1)['title']

def random_recommendation():
    movies = qualified['title']
    selector = random.randint(1, 250)
    return movies.iloc[selector]

def genre_recommendation(genre, percentile=0.60):
    selector = random.randint(1, 50)
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified.sample()['title']


