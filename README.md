# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

**Deadline**: Sunday, Feb 23th 11:59 pm PST

---

## Overview

Build a **content-based recommendation system** that, given a **short text description** of a user’s preferences, suggests **similar items** (e.g., movies) from a small dataset. This challenge should take about **3 hours**, so keep your solution **simple** yet **functional**.

### Example Use Case

- The user inputs:  
  *"I love thrilling action movies set in space, with a comedic twist."*  
- Your system processes this description (query) and compares it to a dataset of items (e.g., movies with their plot summaries or keywords).  
- You then return the **top 3–5 “closest” matches** to the user.

---

## Delivery

1. **Dataset**  
   - Dataset is publicly available at https://www.kaggle.com/datasets/adikhare/top-100-imdb-movies, it contains the top 100 IMDB Movies 

2. **Approach**  
   - Have used DistilBert + TF-IDF score for each word.
   - TF-IDF score is multiplied with Bert Encoding for each word. This helps in better exact matching and capture semantic meaning.
   - Cosine similarity is used to compare the embedding of search query and description of each movie
   - Every sentence is converted in lower case, stemmed and lemmetize before computing the tfidf score and embeddings

3. **Setup**  
   - My Submission contains both the Jupyter and .py scripts.
   - Create a Conda Environment and running the following commands to create env and install requirements
   - ```
     conda create -n "lumaa" python=3.10.0 ipython
     conda activate lumaa
     pip install -r requirements.txt
     ```

4. **Running**  
   - Run the main.py from terminal by following the code below
   - ```
     python main.py -s "I love movies based on gangster, cartels and crime that have drama and action in it with great storyline"
     ```

5. **Result**  
   - The result generated is shown below, it gives 5 Movie Titles along with the description and similarity score
   - ```
     [nltk_data] Downloading package wordnet to
     [nltk_data]     /Users/ritamupadhyay/nltk_data...
     [nltk_data]   Package wordnet is already up-to-date!
     Search Query: I love movies based on gangster, cartels and crime that have drama and action in it with great storyline
     
     Movie Title : Pulp Fiction,
      Description : The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.,
      Similarity Score: 0.8260810375213623
     
     Movie Title : The Hunt,
      Description : A teacher lives a lonely life, all the while struggling over his son's custody. His life slowly gets better as he finds love and receives good news from his son, but his new luck is about to be brutally shattered by an innocent little lie.,
      Similarity Score: 0.8201543092727661
     
     Movie Title : Cinema Paradiso,
      Description : A filmmaker recalls his childhood when falling in love with the pictures at the cinema of his home village and forms a deep friendship with the cinema's projectionist.,
      Similarity Score: 0.8177715539932251
     
     Movie Title : Fight Club,
      Description : An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.,
      Similarity Score: 0.8097826838493347
     
     Movie Title : City of God,
      Description : In the slums of Rio, two kids' paths diverge as one struggles to become a photographer and the other a kingpin.,
      Similarity Score: 0.8060258030891418
     ```

6. **Video Demo**  
   - (https://drive.google.com/file/d/1llmWFv6Yd2OoTthLpwXPboKrlwxNxkQ7/view?usp=sharing)

7. **Salary Expectation**  
   - I am looking for $2,400 per month, based on a rate of $30 per hour for 20 hours per week.

