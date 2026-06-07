from recommender import (
    ARTIFACT_PATH,
    build_model,
    load_data,
    prepare_training_data,
    save_artifact,
)

MIN_USER_RATINGS = 50
MIN_BOOK_RATINGS = 20


def main():
    data = load_data()
    (
        filtered_df,
        book_matrix,
        sparse_matrix,
        book_info_df,
        book_stats_df,
    ) = prepare_training_data(data, MIN_USER_RATINGS, MIN_BOOK_RATINGS)

    model = build_model(sparse_matrix)
    artifact = {
        "model": model,
        "book_matrix": sparse_matrix,
        "book_titles": list(book_matrix.index),
        "book_info": book_info_df,
        "book_stats": book_stats_df,
        "min_user_ratings": MIN_USER_RATINGS,
        "min_book_ratings": MIN_BOOK_RATINGS,
        "filtered_counts": {
            "users": int(filtered_df["userID"].nunique()),
            "books": int(filtered_df["bookTitle"].nunique()),
            "ratings": int(len(filtered_df)),
        },
    }
    save_artifact(artifact)
    print(f"Saved artifact to {ARTIFACT_PATH}")
    print(
        f"Training set: {filtered_df['userID'].nunique():,} users, "
        f"{filtered_df['bookTitle'].nunique():,} books, {len(filtered_df):,} ratings"
    )


if __name__ == "__main__":
    main()
