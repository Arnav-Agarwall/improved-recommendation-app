import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():    
    df = pd.read_csv('tmdb_5000_movies.csv')

    # Preprocess the dataset
    def extract_names(data):
        try:
            data = ast.literal_eval(data)
            return [item['name'] for item in data]
        except (ValueError, SyntaxError):
            return []

    df['genres_processed'] = df['genres'].apply(extract_names)
    df['keywords_processed'] = df['keywords'].apply(extract_names)
    df['combined_features'] = (
        df['genres_processed'].apply(lambda x: ' '.join(x)) + ' ' +
        df['keywords_processed'].apply(lambda x: ' '.join(x)) + ' ' +
        df['overview'].fillna('')
    )
    return df

movies_df = load_data()

# Vectorize the combined features
@st.cache_resource
def compute_similarity_matrix(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_similarity_matrix(movies_df)

# Recommendation functions
def recommend_movies(title):
    try:
        idx = movies_df[movies_df['title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:11]]
        return movies_df['title'].iloc[sim_indices].tolist()
    except IndexError:
        return ["Movie not found. Please check the title."]

def recommend_based_on_ratings(user_ratings):
    high_rated_movies = [movie for movie, rating in user_ratings if rating >= 4]
    recommended_movies = set()
    for movie in high_rated_movies:
        recommended_movies.update(recommend_movies(movie))
    rated_movies = {movie for movie, _ in user_ratings}
    final_recommendations = recommended_movies - rated_movies
    return list(final_recommendations)

# Streamlit app
st.title("🎥 Movie Recommendation System")

# Tabs for the app
tab1, tab2 = st.tabs(["Single Movie Recommendation", "User Ratings Recommendation"])

# Tab 1: Single Movie Recommendation
with tab1:
    st.subheader("Get Recommendations Based on a Single Movie")
    movie_title = st.text_input("Enter a movie title:", "")
    if st.button("Recommend Movies"):
        if movie_title:
            recommendations = recommend_movies(movie_title)
            st.write("### Recommended Movies:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.error("Please enter a movie title.")

# Tab 2: User Ratings Recommendation
with tab2:
    st.subheader("Get Recommendations Based on Your Ratings")
    user_ratings = []
    with st.form("ratings_form"):
        num_movies = st.number_input("How many movies do you want to rate?", min_value=1, step=1, value=3)
        for i in range(num_movies):
            title = st.text_input(f"Movie {i + 1} Title:", key=f"title_{i}")
            rating = st.slider(f"Rating for Movie {i + 1}:", min_value=0, max_value=5, step=1, key=f"rating_{i}")
            if title:
                user_ratings.append((title, rating))
        submitted = st.form_submit_button("Get Recommendations")
        if submitted:
            recommendations = recommend_based_on_ratings(user_ratings)
            if recommendations:
                st.write("### Recommended Movies:")
                for rec in recommendations:
                    st.write(f"- {rec}")
            else:
                st.warning("No recommendations found. Please check your input.")

