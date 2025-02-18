from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

# get the tfidf score for each word and return the in form of dict
def get_word2tfidf(df,search_query):
    tfidf_matrix = tfidf.fit_transform(df['description_final'].to_list()+[search_query])
    word2tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    return word2tfidf