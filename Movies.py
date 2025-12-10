#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Create user-movie matrix
user_movie_matrix = ratings.pivot_table(
    index="userId", columns="movieId", values="rating"
).fillna(0)

# -------------------------------
# STEP 2: COMPUTE USER SIMILARITY
# -------------------------------

user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# -------------------------------
# STEP 3: RECOMMENDATION FUNCTION
# -------------------------------

def recommend_movies(user_id, top_n=5):

    # 1. Get users with high similarity
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # 2. Weighted rating formula
    weighted_ratings = user_movie_matrix.T.dot(similar_users)
    similarity_sum = similar_users.sum()

    scores = weighted_ratings / similarity_sum

    # 3. Remove movies already watched
    user_rated = user_movie_matrix.loc[user_id] > 0
    scores = scores[~user_rated]

    # 4. Top movie IDs
    top_movie_ids = scores.sort_values(ascending=False).head(top_n).index

    # 5. Return movie names
    return movies[movies["movieId"].isin(top_movie_ids)]["title"].tolist()

# -------------------------------
# STEP 4: TEST RECOMMENDATION
# -------------------------------

print("Recommended movies for user 1:")
print(recommend_movies(1, top_n=5))


# In[ ]:




