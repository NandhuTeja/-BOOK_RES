import streamlit as st
import pickle
import numpy as np
import pandas as pd
import urllib.parse

# Load data
book_names = pickle.load(open('book_names.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

books = pd.read_csv('BX-Books.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
books = books[['Book-Title', 'Image-URL-L']]

# Image + Link Helper
def fetch_book_image(book_title):
    try:
        return books[books['Book-Title'] == book_title]['Image-URL-L'].values[0]
    except:
        return "https://via.placeholder.com/150"

def get_amazon_link(book_title):
    encoded = urllib.parse.quote_plus(book_title)
    return f"https://www.amazon.in/s?k={encoded}"

# --- Custom CSS for pro look ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #101010;
        color: #f2f2f2;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h3, h5 {
        font-family: 'Segoe UI', sans-serif;
    }
    .book-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        transition: 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .book-container:hover {
        transform: scale(1.03);
        border: 1px solid #0cf;
        box-shadow: 0 0 15px #0cf;
    }
    .amazon-button {
        background-color: #FF9900;
        color: white;
        padding: 8px 14px;
        border-radius: 10px;
        font-weight: bold;
        text-decoration: none;
        display: inline-block;
        margin-top: 10px;
    }
    .amazon-button:hover {
        background-color: #e88b00;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align:center; color:#FFD700;'>📚 AI Book Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Your personalized reading assistant powered by AI 🔍</p>", unsafe_allow_html=True)

# --- Book Selection ---
selected_book_name = st.selectbox(
    "Choose a book you like 👇",
    book_names,
    key="select_book"
)

# --- Recommend Button ---
if st.button("🔍 Recommend Books"):
    book_index = np.where(book_pivot.index == selected_book_name)[0][0]
    distances, suggestions = model.kneighbors(
        book_pivot.iloc[book_index, :].values.reshape(1, -1), n_neighbors=6
    )

    st.markdown("---")
    st.markdown("<h3 style='color:#00FFFF;'>📘 Recommended for You:</h3>", unsafe_allow_html=True)

    cols = st.columns(5)

    for idx, col in enumerate(cols):
        if idx < len(suggestions[0]) - 1:
            book_title = book_pivot.index[suggestions[0][idx + 1]]
            image_url = fetch_book_image(book_title)
            amazon_link = get_amazon_link(book_title)

            with col:
                st.markdown(f"""
                    <div class="book-container">
                        <img src="{image_url}" width="120" style="border-radius: 8px;" />
                        <h5 style="color:#ffffff; font-size:14px;">{book_title}</h5>
                        <a href="{amazon_link}" class="amazon-button" target="_blank">🛒 View on Amazon</a>
                    </div>
                """, unsafe_allow_html=True)
