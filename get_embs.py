import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tf_idf import get_word2tfidf
from data_load import get_preprocessed_df
from preprocess_search_query import get_preprocessed_search_query

bert_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
model = AutoModel.from_pretrained(bert_model_name)


## Get Distil Bert Embeddings and combine tfidf with bert embeddings
def sentence_embedding(sentence, model, tokenizer, word2tfidf):
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.encode(sentence, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(input_ids)
    
    # Ignore [CLS] and [SEP]
    word_embeddings = outputs.last_hidden_state.squeeze(0)[1:-1]  
    embeddings = []
    
    for i, word in enumerate(tokens):
        if word in word2tfidf:  
            weight = word2tfidf.get(word, 1.0)
            embeddings.append(word_embeddings[i] * weight)
    
    if embeddings:
        return torch.mean(torch.stack(embeddings), dim=0).numpy()
    else:
        return np.zeros(model.config.hidden_size)
    
## gets the preprocessed and transformed dataframe
## preprocesses the search query
## gets the tfidf score for each word
## calculates the weighted matrix with distil bert and tfidf score for each word
## returns the document embeddings, query embeddings and preprocessed dataframe
def get_embeddings(search_query):
    df = get_preprocessed_df()
    search_query = get_preprocessed_search_query(search_query)
    word2tfidf = get_word2tfidf(df,search_query)
    document_embeddings = np.array([sentence_embedding(sent, model, tokenizer, word2tfidf) for sent in df['description_final'].to_list()])
    query_embeddings = np.array([sentence_embedding(search_query, model, tokenizer, word2tfidf)])
    return document_embeddings, query_embeddings, df