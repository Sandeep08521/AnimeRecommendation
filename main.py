import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    else:
        text = str(text)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text


# Load anime data
@st.cache_resource
def load_anime_data(file_path):
    anime_data = pd.read_csv(file_path, encoding='latin1')  # Specify the encoding parameter
    anime_data["Synopsis_processed"] = anime_data["summary"].apply(preprocess_text)
    return anime_data


# Function to calculate cosine similarity
@st.cache_data
def calculate_similarity(anime_data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(anime_data['Synopsis_processed'])
    anime_similarities = cosine_similarity(tfidf_features, tfidf_features)
    return anime_similarities


# Function to recommend similar anime
@st.cache_data
def recommend_anime(anime_name, anime_data, anime_similarities):
    index = anime_data[anime_data["title"] == anime_name].index[0]
    similar_anime_indices = anime_similarities[index].argsort()[::-1][1:6]
    recommended_anime_images = anime_data.iloc[similar_anime_indices]["image_path"].tolist()
    recommended_anime_names = anime_data.iloc[similar_anime_indices]["title"].tolist()
    recommended_anime_summaries = anime_data.iloc[similar_anime_indices]["summary"].tolist()
    return recommended_anime_images, recommended_anime_names, recommended_anime_summaries


# Main Streamlit app
def main():
    # Set background image
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIzLTExL3NtYWxsZGVzaWduY29tcGFueTAxX2FfY3V0ZV9pbGx1c3RyYXRpb25fb2ZfY29tcGFjdF9jaXR5X3ZpZXdfbV8yOWUwZDM5YS02MjI3LTQ4NjktODUwNi0wODIwMjFjN2IxODlfMS5qcGc.jpg") no-repeat center center;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Anime Recommendation')

    # Load anime data
    file_path = r"anime_rec.csv"  # Replace with your actual file path
    anime_data = load_anime_data(file_path)

    # Calculate cosine similarity
    anime_similarities = calculate_similarity(anime_data)

    # Select anime
    selected_anime = st.selectbox('Select an anime:', anime_data['title'])

    # Button to trigger recommendation
    if st.button('Recommend'):
        images, names, summaries = recommend_anime(selected_anime, anime_data, anime_similarities)
        st.success('Recommendations generated!')

        # Display recommendations
        for image, name, summary in zip(images, names, summaries):
            if image:
                st.image(image, caption=name, width=400)  # Increase the image width
            st.subheader(name)  # Display name
            st.write(summary)  # Display summary


if __name__ == '__main__':
    main()
