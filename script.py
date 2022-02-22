#' -*- coding: utf-8 -*-
#' 
#' Criado em fev/2022
#'==========================================================================================================
#' 
#' Função de recomendação de filmes.
#' 
#'==========================================================================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text  import  TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel

# Importar dataset
metadata = pd.read_csv('data/movie_overviews.csv').dropna()
metadata.head()

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

def get_recommendations(title, cosine_sim, indices):
    # Obter o índice do filme que corresponde ao título
    idx = indices[title]
    # Obter o score de similaridade
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Classificar os filmes com base no score de similaridade
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Obter o score dos 10 filmes mais semelhantes
    sim_scores = sim_scores[1:11]
    # Obter os índices de filmes
    movie_indices = [i[0] for i in sim_scores]
    # Retorna os 10 filmes mais parecidos
    return metadata['title'].iloc[movie_indices]  
  
movie_plots = metadata['overview']

tfidf = TfidfVectorizer(stop_words='english')

# Construir a matriz TF-IDF
tfidf_matrix = tfidf.fit_transform(movie_plots)

# Gerar a matriz de similaridade de cosseno 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Gerar recomendações de filmes
print(get_recommendations("High Plains Drifter", cosine_sim, indices))
