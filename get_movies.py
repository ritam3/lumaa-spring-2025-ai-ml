import numpy as np
import pandas as pd

top_k_matches = 5

## function to calculate cosine distance, larger the value smaller the angle and hence greater the similarity
def similarity(x,y):
    cos = np.dot(x,y)
    x_mag = np.linalg.norm(x)
    y_mag = np.linalg.norm(y)
    if x_mag==0:
        return "Please elaborate on what kind of movies you like"
    res = cos/(x_mag*y_mag)
    return res.item()

## function to calculate the cosine distance between search query and each sentence in corpus
## return the top 5 similarity matches in the form of dataframe
def get_movies(document_embeddings, query_embeddings, df):
    res = []
    for emb in document_embeddings:
        sim = similarity(query_embeddings,emb)
        if type(sim)==str:
            return sim
        res.append(sim)
    df['similarity'] = res
    sim_arg_k = np.argsort(res)[-top_k_matches:][::-1]
    return df.iloc[sim_arg_k][['title','similarity','description']]