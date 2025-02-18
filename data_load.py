from get_data import get_data
from data_transform import preprocess, add_date_genre
from dotenv import load_dotenv
import os
load_dotenv()


data_path = os.getenv("path")

df = get_data(data_path)

## adds the date and genre into description
## stems and lemmetize the description to make final description
def get_preprocessed_df():
    df['description_2'] = df.apply(lambda x: add_date_genre(x),axis=1)
    df['description_final'] = df.apply(lambda x: preprocess(x['description_2']),axis=1)
    return df