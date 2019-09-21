import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data1 = pd.read_csv('tmdb_5000_credits.csv')
data2 = pd.read_csv('tmdb_5000_movies.csv')

data1.columns = ['id', 'title', 'cast', 'crew']
data2 = data2.merge(data1 ,on = 'id')

data3 = data2.iloc[0:4,:].values
data3 = pd.DataFrame(data3)

#Demographic filtering
C = data2['vote_average'].mean()
m = data2['vote_count'].quantile(0.9)
q_movies = data2[data2['vote_count']>=m]
q_movies.shape()

def weighted_ratio(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_ratio, axis = 1)

q_movies = q_movies.sort_values('score', ascending=False)

q = q_movies[['title_y','vote_count','vote_average','score']].iloc[0:9].values
q = pd.DataFrame(q)

pop = q_movies.sort_values('popularity', ascending = False)

plt.figure(figsize=(12,4))
plt.barh(pop['title_y'].iloc[0:10].values, pop['popularity'].iloc[0:10].values, color = 'skyblue', align = 'center')
plt.xlabel('Popularity')
plt.title('Popular movies')
plt.show()


# CONTENT BASED FILTERING
data2['overview'].isnull().value_counts()
data2['overview'] = data2['overview'].fillna(' ')

from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer(stop_words = 'english')
tfid_matrix = tfid.fit_transform(data2['overview'])
tfid_matrix.shape()

t = tfid_matrix.toarray() #returns a  matrix

tfid.get_feature_names()

p = pd.DataFrame(tfid_matrix.toarray(), columns = tfid.get_feature_names())

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfid_matrix, tfid_matrix)

indices = pd.Series(data2.index, index = data2['title_y']).drop_duplicates()

def get_recommendation(title_y, cosine_sim = cosine_sim):
    # Get the index of the movie that matches the title
    index = indices[title_y]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[index]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return data2['title_y'].iloc[movie_indices]

get_recommendation('The Dark Knight Rises')

get_recommendation('The Avengers')

# Credits, Genres and Keywords Based Recommender

# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast','crew','keywords','genres']

for feature in features:
    data2[feature] = data2[feature].apply(literal_eval) #convert list of python dictionaries to objects 
    
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
             return i["name"]
    return np.nan

def get_list(x):
   if isinstance(x, list): #if x is a list
        names = [i['name'] for i in x]
        if len(names)>3 :
           names = names[:3]
        return names 
    
   else :
       return []

data2['director'] = data2['crew'].apply(get_director)

features = ['cast','keywords','genres']

for feature in features:
    data2[feature] = data2[feature].apply(get_list)
        
r = data2[['title_y', 'cast', 'director', 'keywords', 'genres']].iloc[0:3,:].values   
r = pd.DataFrame(r)    
        
'''The next step would be to convert the names and keyword instances into lowercase and strip all
 the spaces between them. This is done so that our vectorizer doesn't count the Johnny of "Johnny
 Depp" and "Johnny Galecki" as the same.'''
 
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ","")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ",""))
        else:
            return ' '
 
features = ['cast','keywords','director','genres']
 
for feature in features:
    data2[feature] = data2[feature].apply(clean_data)

''' "metadata soup", which is a string that contains all the metadata that we want to feed to our
 vectorizer (namely actors, director and keywords)'''
 
def create_soup(x):
   return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
 
 data2['soup'] = data2.apply(create_soup, axis = 1)

from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer just counts the word frequencies
count = CountVectorizer(stop_words = 'english')
count_matrix = count.fit_transform(data2['soup'])

#Cosine similarity is a metric used to determine how similar the documents are irrespective of 
#their size

sparse_matrix = count_matrix.todense() #returns a matrix

d = pd.DataFrame(sparse_matrix, columns = count.get_feature_names())

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

data2 = data2.reset_index()
indices = pd.Series(data2.index, index = data2['title_y'])

get_recommendation('The Dark Knight Rises', cosine_sim2)

get_recommendation('The Godfather', cosine_sim2)

#Collaborative Filtering

# 1) User based filtering
ratings = pd.read_csv('ratings_small.csv')

from surprise import Reader, evaluate, Dataset, SVD #surprise contains lot of dataset related to ratings which are small and easy to use
reader = Reader()

data = Dataset.load_from_file(ratings[['userID','movieID','rating']], reader)
data.split(n_folds = 5)

svd  = np.linalg.SVD()
evaluate(svd, data, measures = ['RMSE','MAE'])

trainset = data.build_full_trainset()
svd.fit(trainset)

ratings(ratings["userId"]==1)

svd.predict(2,303,4)


