import argparse
import pandas as pd
from get_embs import get_embeddings
from get_movies import get_movies

## Display the Movie titles with Description and Similarity Scores
def display_movies(df):
    for _,row in df.iterrows():
        print(f"Movie Title : {row['title']},\n Description : {row['description']},\n Similarity Score: {row['similarity']}\n")

def main():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    
    # Adding arguments
    parser.add_argument("-s", "--search", type=str, help="Search Query By User")
    
    # Parsing arguments
    args = parser.parse_args()
    
    # Printing the received arguments
    print(f"Search Query: {args.search}\n")

    #Get the embeddings for the corpus and search query
    doc_emb, query_embs, df = get_embeddings(args.search)

    #Get the DataFrame for top 5 matches or the Message string asking to elaborate
    df_result = get_movies(doc_emb,query_embs, df)

    if type(df_result)==str:
        print(df_result)
        return

    display_movies(df_result)

if __name__ == "__main__":
    main()