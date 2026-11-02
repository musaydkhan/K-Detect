import pandas as pd
import re
import unicodedata
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parents[1]

RAID_FILE = (
    BASE_DIR / "datasets" / "processed" / "raid_abstracts.csv"
)

SCIHRA_FILE = (
    BASE_DIR / "datasets" / "processed" / "scihra_processed.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "evaluation" / "raid_scihra_near_duplicates.csv"
)


def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():

    print("Loading human abstracts...")

    raid = pd.read_csv(RAID_FILE)
    scihra = pd.read_csv(SCIHRA_FILE)

    raid = raid[raid["model"] == "human"].copy()

    scihra = scihra[
        scihra["class_name"] == "Human-written"
    ].copy()

    raid["normalized"] = raid["generation"].apply(
        normalize_text
    )

    scihra["normalized"] = scihra["text"].apply(
        normalize_text
    )

    raid = raid[raid["normalized"] != ""].reset_index(drop=True)
    scihra = scihra[scihra["normalized"] != ""].reset_index(drop=True)

    print("RAID abstracts:", len(raid))
    print("SciHRA human abstracts:", len(scihra))

    print("\nBuilding character TF-IDF index...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=300000,
        sublinear_tf=True
    )

    combined = (
        raid["normalized"].tolist()
        + scihra["normalized"].tolist()
    )

    matrix = vectorizer.fit_transform(combined)

    raid_matrix = matrix[:len(raid)]
    scihra_matrix = matrix[len(raid):]

    print("\nSearching nearest SciHRA matches...")

    search = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )

    search.fit(scihra_matrix)

    distances, indices = search.kneighbors(raid_matrix)

    similarities = 1 - distances.flatten()

    results = []

    for i, similarity in enumerate(similarities):

        j = indices[i][0]

        results.append({
            "raid_source_id": raid.iloc[i]["source_id"],
            "raid_title": raid.iloc[i]["title"],
            "scihra_article_id": scihra.iloc[j]["article_id"],
            "scihra_domain": scihra.iloc[j]["domain"],
            "scihra_source": scihra.iloc[j]["source"],
            "similarity": round(float(similarity), 4)
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "similarity",
        ascending=False
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== SIMILARITY SUMMARY =====")

    print("Maximum similarity:", result_df["similarity"].max())

    for threshold in [0.70, 0.80, 0.90, 0.95]:

        count = (
            result_df["similarity"] >= threshold
        ).sum()

        print(
            f"Similarity >= {threshold}: {count}"
        )

    print("\n===== TOP 15 POTENTIAL MATCHES =====")

    print(
        result_df.head(15).to_string(index=False)
    )

    print("\nSaved results:", OUTPUT_FILE)


if __name__ == "__main__":
    main()