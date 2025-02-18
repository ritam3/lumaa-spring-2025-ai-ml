from data_transform import preprocess

# Preprocess i.e. stemm and lemmetize the search query
def get_preprocessed_search_query(search_query):
    return preprocess(search_query)