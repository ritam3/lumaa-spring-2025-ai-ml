import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

## function to stem and lemmetize a given sentence
def preprocess(text):
    words = text.lower().split()
    stemmed_words = [stemmer.stem(word) for word in words]  # Apply stemming first
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]  # Apply lemmatization
    return " ".join(lemmatized_words)

## Idea is to add the genre and date of release in the description
def add_date_genre(x):
    release_date = " The movie was released in the year "
    genre = " The genre is of "
    return x['description']+release_date+str(x['year'])+'.'+genre+' and '.join(eval(x['genre']))

