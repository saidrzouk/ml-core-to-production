import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.recommender import (
    ARTIFACT_PATH,
    load_artifact,
    recommend_similar_books,
)


def format_year(value):
    try:
        if value is None or value != value:
            return "Unknown"
        return str(int(value))
    except (TypeError, ValueError):
        return "Unknown"


def format_number(value):
    try:
        if value is None or value != value:
            return "Unknown"
        return f"{int(value)}"
    except (TypeError, ValueError):
        return "Unknown"


def format_rating(value):
    try:
        if value is None or value != value:
            return "Unknown"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "Unknown"


def show_book_card(row):
    image_url = row.get("imageUrlM")
    columns = st.columns([1, 4])

    with columns[0]:
        if isinstance(image_url, str) and image_url.startswith("http"):
            st.image(image_url, width=90)

    with columns[1]:
        st.subheader(row["bookTitle"])
        st.write(f"Author: {row['bookAuthor']}")
        st.write(f"Publisher: {row['publisher']}")
        st.write(f"Year: {format_year(row.get('yearOfPublication'))}")
        st.write(
            f"Average rating: {format_rating(row.get('averageRating'))} "
            f"from {format_number(row.get('numberOfRatings'))} ratings"
        )
        if "similarityScore" in row:
            st.write(f"Similarity score: {row['similarityScore']:.3f}")


st.set_page_config(page_title="Book Recommendation System", layout="wide")

st.title("Book Recommendation System")

if not ARTIFACT_PATH.exists():
    st.error(
        "No trained artifact found yet. Run `python src/train.py` from the project folder first."
    )
    st.stop()


@st.cache_resource
def get_artifact():
    return load_artifact()


artifact = get_artifact()
book_info_df = artifact["book_info"]
book_stats_df = artifact["book_stats"]
filtered_counts = artifact["filtered_counts"]
number_of_recommendations = 10

st.caption(
    f"Using {filtered_counts['users']:,} users, "
    f"{filtered_counts['books']:,} books, and "
    f"{filtered_counts['ratings']:,} explicit ratings."
)

tab_recommend, tab_popular, tab_search = st.tabs(
    ["Similar Books", "Popular Books", "Search"]
)

with tab_recommend:
    selected_book = st.selectbox(
        "Choose a book",
        sorted(artifact["book_titles"]),
    )

    recommendations = recommend_similar_books(selected_book, artifact, n=number_of_recommendations)

    st.write(f"Books similar to **{selected_book}**")
    for _, recommendation in recommendations.iterrows():
        show_book_card(recommendation)
        st.divider()

with tab_popular:
    popular_books = (
        book_stats_df.sort_values("popularityScore", ascending=False)
        .head(number_of_recommendations)
        .merge(
            book_info_df.drop(columns=["averageRating", "numberOfRatings"]),
            on="bookTitle",
            how="left",
        )
    )

    st.write("Most popular books by rating quality and rating volume")
    for _, book in popular_books.iterrows():
        show_book_card(book)
        st.divider()

with tab_search:
    keyword = st.text_input("Search book titles", value="Harry Potter")
    if keyword:
        matches = book_info_df[
            book_info_df["bookTitle"].str.contains(keyword, case=False, na=False)
        ].sort_values("numberOfRatings", ascending=False)

        st.write(f"Found {len(matches):,} matches")
        st.dataframe(
            matches[
                [
                    "bookTitle",
                    "bookAuthor",
                    "publisher",
                    "yearOfPublication",
                    "averageRating",
                    "numberOfRatings",
                ]
            ].head(50),
            use_container_width=True,
        )
