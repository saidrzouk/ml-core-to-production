import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from src.config import ARTIFACT_PATH, DATA_DIR

BOOK_COLUMNS = [
    "ISBN",
    "bookTitle",
    "bookAuthor",
    "yearOfPublication",
    "publisher",
    "imageUrlS",
    "imageUrlM",
    "imageUrlL",
]
USER_COLUMNS = ["userID", "location", "age"]
RATING_COLUMNS = ["userID", "ISBN", "bookRating"]


def assert_columns(df, required_columns, name):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def assert_non_empty(df, name):
    if df.empty:
        raise ValueError(f"{name} is empty after validation and filtering")


def validate_artifact(artifact):
    required_keys = {
        "model",
        "book_matrix",
        "book_titles",
        "book_info",
        "book_stats",
        "filtered_counts",
    }
    missing = required_keys - set(artifact)
    if missing:
        raise ValueError(f"Artifact is missing keys: {sorted(missing)}")

    assert_columns(
        artifact["book_info"],
        ["bookTitle", "bookAuthor", "publisher", "yearOfPublication", "averageRating", "numberOfRatings", "imageUrlM"],
        "artifact['book_info']",
    )
    assert_columns(
        artifact["book_stats"],
        ["bookTitle", "averageRating", "numberOfRatings", "popularityScore"],
        "artifact['book_stats']",
    )

    if not isinstance(artifact["book_titles"], list) or not artifact["book_titles"]:
        raise ValueError("artifact['book_titles'] must be a non-empty list")
    if "users" not in artifact["filtered_counts"]:
        raise ValueError("artifact['filtered_counts'] must include 'users'")
    if "books" not in artifact["filtered_counts"]:
        raise ValueError("artifact['filtered_counts'] must include 'books'")
    if "ratings" not in artifact["filtered_counts"]:
        raise ValueError("artifact['filtered_counts'] must include 'ratings'")


def load_data():
    books = pd.read_csv(
        DATA_DIR / "BX_Books.csv",
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
    )
    if books.shape[1] != len(BOOK_COLUMNS):
        raise ValueError(
            f"BX_Books.csv must have {len(BOOK_COLUMNS)} columns, found {books.shape[1]}"
        )
    users = pd.read_csv(
        DATA_DIR / "BX-Users.csv",
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
    )
    if users.shape[1] != len(USER_COLUMNS):
        raise ValueError(
            f"BX-Users.csv must have {len(USER_COLUMNS)} columns, found {users.shape[1]}"
        )
    ratings = pd.read_csv(
        DATA_DIR / "BX-Book-Ratings.csv",
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
    )
    if ratings.shape[1] != len(RATING_COLUMNS):
        raise ValueError(
            f"BX-Book-Ratings.csv must have {len(RATING_COLUMNS)} columns, found {ratings.shape[1]}"
        )

    books.columns = BOOK_COLUMNS
    users.columns = USER_COLUMNS
    ratings.columns = RATING_COLUMNS

    assert_columns(books, BOOK_COLUMNS, "BX_Books.csv")
    assert_columns(users, USER_COLUMNS, "BX-Users.csv")
    assert_columns(ratings, RATING_COLUMNS, "BX-Book-Ratings.csv")

    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    books["yearOfPublication"] = pd.to_numeric(
        books["yearOfPublication"],
        errors="coerce",
    )
    ratings["bookRating"] = pd.to_numeric(ratings["bookRating"], errors="coerce")

    users = users[users["age"].between(1, 100) | users["age"].isna()].copy()
    books = books[books["yearOfPublication"].between(1800, 2026)].copy()
    books = books.dropna(
        subset=["ISBN", "bookTitle", "bookAuthor", "publisher"]
    ).copy()
    ratings = ratings[ratings["bookRating"].between(1, 10)].copy()

    merged = (
        ratings.merge(users, on="userID", how="inner")
        .merge(books, on="ISBN", how="inner")
    )

    assert_non_empty(merged, "merged book recommendation data")
    return merged


def prepare_training_data(data, min_user_ratings, min_book_ratings):
    user_counts = data["userID"].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    filtered = data[data["userID"].isin(active_users)].copy()

    book_counts = filtered["bookTitle"].value_counts()
    popular_books = book_counts[book_counts >= min_book_ratings].index
    filtered = filtered[filtered["bookTitle"].isin(popular_books)].copy()

    assert_non_empty(filtered, "filtered training data")

    book_user_matrix = filtered.pivot_table(
        index="bookTitle",
        columns="userID",
        values="bookRating",
    ).fillna(0)
    if book_user_matrix.empty:
        raise ValueError("book-user matrix is empty after filtering")
    sparse_matrix = csr_matrix(book_user_matrix.values)

    book_info = (
        filtered.groupby("bookTitle")
        .agg(
            bookAuthor=("bookAuthor", "first"),
            publisher=("publisher", "first"),
            yearOfPublication=("yearOfPublication", "first"),
            averageRating=("bookRating", "mean"),
            numberOfRatings=("bookRating", "count"),
            imageUrlM=("imageUrlM", "first"),
        )
        .reset_index()
    )

    book_stats = (
        filtered.groupby("bookTitle")["bookRating"]
        .agg(["mean", "count"])
        .reset_index()
    )
    book_stats.columns = ["bookTitle", "averageRating", "numberOfRatings"]
    book_stats["popularityScore"] = (
        book_stats["averageRating"] * book_stats["numberOfRatings"]
    )

    return filtered, book_user_matrix, sparse_matrix, book_info, book_stats


def build_model(book_matrix):
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(book_matrix)
    return model


def save_artifact(artifact, artifact_path=ARTIFACT_PATH):
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)


def load_artifact(artifact_path=ARTIFACT_PATH):
    artifact = joblib.load(artifact_path)
    validate_artifact(artifact)
    return artifact


def recommend_similar_books(book_title, artifact, n=10):
    book_titles = artifact["book_titles"]
    if book_title not in book_titles:
        raise KeyError(f"Unknown book title: {book_title}")

    book_index = book_titles.index(book_title)
    book_matrix = artifact["book_matrix"]
    model = artifact["model"]

    neighbor_count = min(n + 1, len(book_titles))
    distances, indices = model.kneighbors(
        book_matrix[book_index],
        n_neighbors=neighbor_count,
    )
    recommendations = []
    for distance, index in zip(distances.flatten(), indices.flatten()):
        if index == book_index:
            continue
        recommendations.append(
            {
                "bookTitle": book_titles[index],
                "similarityScore": 1 - float(distance),
            }
        )

    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df.merge(artifact["book_info"], on="bookTitle", how="left")
