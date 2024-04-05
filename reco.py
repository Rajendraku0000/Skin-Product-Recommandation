import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV file
df = pd.read_csv("C:/Users/rajen/OneDrive/Desktop/Skin tone product/recommendations.csv", encoding='latin-1')
# print(df)

# Lowercase the "Face Shape" and "Goggles" columns
df["Face Shape"] = df["Face Shape"].str.lower()

# Convert float values in "review" column to string
df["review"] = df["review"].astype(str)

# Concatenate relevant columns into a new column "total"
df["total"] = df["Face Shape"]  + ' ' + df["review"]

# Initialize CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words="english")

# Fit and transform the "total" column
vector = cv.fit_transform(df["total"]).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vector)

def recommend(text):
    global df
    index = None
    # Iterate through each row to find the index where the "Face Shape" or "Goggles" matches the input text
    for i, row in df.iterrows():
        if text in row["Face Shape"]:
            index = i
            break

    if index is None:
        return "No matching recommendations found."

    # Sort distances and get top 5 similar indices
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    indices = [i[0] for i in distances[1:6]]
    
    # Return recommendations
    return df.iloc[indices, [1,2]]

# # Test the recommend function
# a = recommend("Dark-Normal skin tone")
# print(a)
