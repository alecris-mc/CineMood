import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
MOOD_PROFILES = ['Feel-good', 'Romantic', 'Funny', 'Exciting', 'Dark & Suspenseful', 'Emotional', 'Inspiring', 'Thought-provoking']

def recommend_by_sentiment(
    target_sentiment,
    top_n=10
):

    target_sentiment = float(
        np.clip(target_sentiment, -1, 1)
    )

    results = df.copy()

    results["sentiment_similarity"] = (
        1 - abs(
            results["sentiment_compound"]
            - target_sentiment
        ) / 2
    )

    results = results.sort_values(
        ["sentiment_similarity", "weighted_rating"],
        ascending=False
    )

    return results[[
        "title",
        "genres",
        "sentiment_compound",
        "sentiment_similarity",
        "weighted_rating"
    ]].head(top_n)

def filter_movies(
    year_range=None,
    release_period=None,
    min_rating=0,
    min_votes=0,
    runtime_range=None,
    certifications=None,
    languages=None,
    actors=None,
    directors=None,
    genres=None,
    genre_operator="AND",
    excluded_genres=None
):

    results = df.copy()

    # Temporal filters
    if year_range is not None:
        results = results[
            results["year"].between(*year_range)
        ]

    if release_period is not None:
        start, end = release_period

        results = results[
            results["release_date"].between(
                pd.Timestamp(start),
                pd.Timestamp(end)
            )
        ]

    # Quality filters
    results = results[
        (results["vote_average"] >= min_rating)
        & (results["vote_count"] >= min_votes)
    ]

    # Runtime filter
    if runtime_range is not None:
        results = results[
            results["runtime"].between(
                *runtime_range
            )
        ]

    # Certification and language
    if certifications:
        results = results[
            results["certification"].isin(
                certifications
            )
        ]

    if languages:
        results = results[
            results["original_language"].isin(
                languages
            )
        ]

    # Actor preferences
    if actors:
        selected = {
            name.casefold() for name in actors
        }

        results = results[
            results["cast"].apply(
                lambda names: bool(
                    selected & {
                        n.casefold() for n in names
                    }
                )
            )
        ]

    # Director preferences
    if directors:
        selected = {
            name.casefold() for name in directors
        }

        results = results[
            results["directors"].apply(
                lambda names: bool(
                    selected & {
                        n.casefold() for n in names
                    }
                )
            )
        ]

    # Genre logic: AND / OR
    if genres:

        selected_genres = set(genres)

        if genre_operator == "AND":
            results = results[
                results["genres"].apply(
                    lambda x:
                    selected_genres.issubset(set(x))
                )
            ]

        elif genre_operator == "OR":
            results = results[
                results["genres"].apply(
                    lambda x: bool(
                        selected_genres & set(x)
                    )
                )
            ]

        else:
            raise ValueError(
                "Genre operator must be AND or OR."
            )

    # Genre exclusion: NOT
    if excluded_genres:
        excluded = set(excluded_genres)

        results = results[
            results["genres"].apply(
                lambda x: not bool(
                    excluded & set(x)
                )
            )
        ]

    return results.copy()

GENRE_ALIASES = {
    "Action": ["action"],
    "Adventure": ["adventure"],
    "Comedy": ["comedy", "funny", "hilarious"],
    "Romance": ["romance", "romantic"],
    "Horror": ["horror", "scary"],
    "Thriller": ["thriller", "suspense"],
    "Drama": ["drama"],
    "Science Fiction": ["science fiction", "sci-fi"],
    "Fantasy": ["fantasy"],
    "Animation": ["animation", "animated"],
    "Family": ["family"],
    "Crime": ["crime"],
    "Mystery": ["mystery"],
    "Documentary": ["documentary"],
    "History": ["history", "historical"],
    "Music": ["music", "musical"],
    "War": ["war"],
    "Western": ["western"]
}

MOOD_ALIASES = {
    "Feel-good": [
        "feel-good", "feel good", "happy",
        "cheerful", "heartwarming", "comforting"
    ],
    "Romantic": ["romantic", "romance", "love"],
    "Funny": ["funny", "hilarious", "laugh"],
    "Exciting": ["exciting", "adrenaline", "action-packed"],
    "Dark & Suspenseful": [
        "dark", "suspenseful", "scary"
    ],
    "Emotional": ["emotional", "sad", "moving"],
    "Inspiring": ["inspiring", "motivational", "uplifting"],
    "Thought-provoking": [
        "thought-provoking", "intellectual",
        "philosophical", "mind-bending"
    ]
}

def contains_phrase(text, phrase):
    return bool(re.search(
        r"\b" + re.escape(phrase) + r"\b",
        text
    ))

def parse_query(query):

    text = query.lower().strip()

    preferences = {
        "genres": [],
        "excluded_genres": [],
        "genre_operator": "AND",
        "mood": None,
        "runtime_range": None,
        "min_rating": 0,
        "year_range": None
    }

    positive_text = text

    # Detect explicit genre exclusions
    for genre, aliases in GENRE_ALIASES.items():

        for alias in aliases:

            pattern = (
                r"\b(?:no|not|without|exclude|avoid)"
                r"\s+(?:any\s+)?"
                + re.escape(alias)
                + r"\b"
            )

            if re.search(pattern, text):
                preferences["excluded_genres"].append(
                    genre
                )

                positive_text = re.sub(
                    pattern, " ", positive_text
                )

    # Detect requested genres
    for genre, aliases in GENRE_ALIASES.items():

        if genre not in preferences["excluded_genres"]:

            if any(
                contains_phrase(positive_text, alias)
                for alias in aliases
            ):
                preferences["genres"].append(genre)

    # Detect genre alternatives
    if contains_phrase(positive_text, "or"):
        preferences["genre_operator"] = "OR"

    # Identify strongest matching mood
    mood_matches = {
        mood: sum(
            contains_phrase(positive_text, alias)
            for alias in aliases
        )
        for mood, aliases in MOOD_ALIASES.items()
    }

    if max(mood_matches.values()) > 0:
        preferences["mood"] = max(
            mood_matches, key=mood_matches.get
        )

    # Detect maximum runtime
    runtime_match = re.search(
        r"(under|less than|at most|up to|maximum)"
        r"\s+(\d+(?:\.\d+)?)"
        r"\s*(minutes?|mins?|hours?|hrs?)\b",
        text
    )

    if runtime_match:

        operator = runtime_match.group(1)
        duration = float(runtime_match.group(2))
        unit = runtime_match.group(3)

        if unit.startswith(("hour", "hr")):
            duration *= 60

        if operator in ["under", "less than"]:
            duration = int(np.ceil(duration)) - 1
        else:
            duration = int(duration)

        preferences["runtime_range"] = (
            1, duration
        )

    # Detect minimum rating
    rating_match = re.search(
        r"(?:rated|rating)\s*"
        r"(?:above|over|at least|of|>=)?"
        r"\s*(\d+(?:\.\d+)?)",
        text
    )

    if rating_match:
        rating = float(rating_match.group(1))

        if 0 <= rating <= 10:
            preferences["min_rating"] = rating

    # Detect decade
    decade_match = re.search(
        r"\b((?:19|20)\d0)s\b",
        text
    )

    if decade_match:
        decade = int(decade_match.group(1))

        preferences["year_range"] = (
            decade, decade + 9
        )

    # Detect explicit year range
    year_match = re.search(
        r"\b((?:19|20)\d{2})\s*"
        r"(?:to|-)\s*((?:19|20)\d{2})\b",
        text
    )

    if year_match:
        preferences["year_range"] = tuple(
            sorted(map(int, year_match.groups()))
        )

    preferences["excluded_genres"] = list(
        dict.fromkeys(
            preferences["excluded_genres"]
        )
    )

    return preferences

def hybrid_recommend(
    query="",
    mood=None,
    reference_title=None,
    top_n=10,
    filters=None,
    weights=(0.50, 0.35, 0.15)
):

    filters = filters or {}

    candidates = filter_movies(**filters)

    if candidates.empty:
        return candidates

    # Initialize component scores
    content_scores = np.zeros(len(df))
    mood_scores = np.zeros(len(df))

    active_weights = np.array(
        weights, dtype=float
    )

    if (
        active_weights.shape != (3,)
        or not np.isfinite(active_weights).all()
        or (active_weights < 0).any()
    ):
        raise ValueError("Invalid model weights.")

    content_available = False

    # Content similarity from a reference movie
    if reference_title:

        matches = df.index[
            df["title"].str.casefold()
            == reference_title.strip().casefold()
        ].tolist()

        if not matches:
            raise ValueError(
                "Reference movie not found."
            )

        reference_index = matches[0]

        content_scores = similarity_matrix[
            reference_index
        ].copy()

        candidates = candidates.drop(
            index=reference_index,
            errors="ignore"
        )

        content_available = True

    # Content similarity from natural language
    if query.strip():

        query_vector = tfidf.transform([query])

        if query_vector.nnz > 0:

            query_scores = cosine_similarity(
                query_vector, tfidf_matrix
            ).flatten()

            if content_available:
                content_scores = (
                    0.60 * content_scores
                    + 0.40 * query_scores
                )
            else:
                content_scores = query_scores

            content_available = True

    if not content_available:
        active_weights[0] = 0

    # Mood compatibility
    if mood in MOOD_PROFILES:

        mood_scores = df[
            f"mood_{mood}"
        ].to_numpy()

    else:
        active_weights[1] = 0

    # Normalize active weights
    if active_weights.sum() == 0:
        active_weights[2] = 1

    active_weights /= active_weights.sum()

    # Quality score
    quality_scores = (
        df["weighted_rating"]
        .fillna(0)
        .to_numpy() / 10
    )

    # Apply scores to filtered candidates
    indices = candidates.index.to_numpy()

    candidates["content_score"] = (
        content_scores[indices]
    )

    candidates["mood_score"] = (
        mood_scores[indices]
    )

    candidates["quality_score"] = (
        quality_scores[indices]
    )

    candidates["hybrid_score"] = (
        active_weights[0]
        * candidates["content_score"]
        + active_weights[1]
        * candidates["mood_score"]
        + active_weights[2]
        * candidates["quality_score"]
    )

    candidates["match_score"] = (
        candidates["hybrid_score"] * 100
    ).round(2)

    candidates["content_weight"] = active_weights[0]
    candidates["mood_weight"] = active_weights[1]
    candidates["quality_weight"] = active_weights[2]

    return candidates.sort_values(
        ["hybrid_score", "weighted_rating"],
        ascending=False
    ).head(top_n).copy()

def explain_recommendation(
    row,
    mood=None,
    reference_title=None,
    requested_genres=None
):

    reasons = []

    # Explain genre compatibility
    if requested_genres:

        shared_genres = sorted(
            set(row["genres"])
            & set(requested_genres)
        )

        if shared_genres:
            reasons.append(
                "Matches your preferred genres: "
                + ", ".join(shared_genres)
            )

    # Explain content similarity
    if row["content_weight"] > 0:

        score = row["content_score"] * 100

        if reference_title:
            reasons.append(
                f"Content similarity to "
                f"{reference_title}: {score:.1f}%"
            )
        else:
            reasons.append(
                f"Text similarity to your query: "
                f"{score:.1f}%"
            )

    # Explain emotional compatibility
    if mood and row["mood_weight"] > 0:

        score = row["mood_score"] * 100

        reasons.append(
            f"{mood} mood compatibility: "
            f"{score:.1f}/100"
        )

    # Explain movie quality
    reasons.append(
        f"TMDB rating: "
        f"{row['vote_average']:.1f}/10 "
        f"from {row['vote_count']:,} votes"
    )

    # Explain duration
    if pd.notna(row["runtime"]):

        reasons.append(
            f"Runtime: {int(row['runtime'])} minutes"
        )

    # Explain the largest ranking contribution
    contributions = {
        "Content similarity":
            row["content_score"]
            * row["content_weight"],

        "Mood compatibility":
            row["mood_score"]
            * row["mood_weight"],

        "Weighted rating":
            row["quality_score"]
            * row["quality_weight"]
    }

    main_factor = max(
        contributions,
        key=contributions.get
    )

    reasons.append(
        f"Main ranking factor: {main_factor}"
    )

    return " | ".join(reasons)

original_parse_query = parse_query

def parse_query(query):

    preferences = original_parse_query(query)

    text = query.lower().strip()

    # Distinguish strict and inclusive rating thresholds
    strict_rating = re.search(
        r"(?:rated|rating)\s+(?:above|over)"
        r"\s+(\d+(?:\.\d+)?)",
        text
    )

    if strict_rating:
        threshold = float(strict_rating.group(1))

        if 0 <= threshold <= 10:
            preferences["min_rating"] = np.nextafter(
                threshold, np.inf
            )

    return preferences