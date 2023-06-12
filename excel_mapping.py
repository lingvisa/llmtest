### An algorithm to find similar columns from two csv files, with the help from the offline interaction through chatGPT prompt engineering
### This works well with the sample data test, though simple enough and without involving LLM in the code

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df1 = pd.read_csv('template.csv')
df2 = pd.read_csv('table_A.csv') # table_B.csv


def compute_similar_columns(df1, df2, threshold):
    similar_columns = []
    # for efficiency, only use top 1000 rows
    df1 = df1.head(100)
    df2 = df2.head(100)

    for column1 in df1.columns:
        values1 = df1[column1].tolist()
        # add column name into the value list
        values1.insert(0, column1)
        for column2 in df2.columns:
            values2 = df2[column2].tolist()
            values2.insert(0, column2)

            # Compute cosine similarity
            similarity = compute_cosine_similarity(values1, values2)

            if similarity > threshold:
                similar_columns.append((column1, column2, similarity))
    return similar_columns

def compute_cosine_similarity(list1, list2):
    # Convert the input lists to strings
    string1 = ' '.join(map(str, list1))
    string2 = ' '.join(map(str, list2))

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer="char")

    # Compute the TF-IDF matrix for the two strings
    tfidf_matrix = vectorizer.fit_transform([string1, string2])

    # Compute the cosine similarity between the two TF-IDF vectors
    similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    # Extract the similarity score from the matrix
    similarity_score = similarity_matrix[0][0]

    return similarity_score

similar_columns = compute_similar_columns(df1, df2, 0.8)
for column1, column2, similarity in similar_columns:
    print(f"Similar columns: {column1} (File 1) and {column2} (File 2), Similarity: {similarity}")

