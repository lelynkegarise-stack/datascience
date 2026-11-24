from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Load FlavorGraph Nodes & Edges
# ---------------------------------------------------------
def normalize_name(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").split()).strip()


def load_flavorgraph(nodes_path: str | Path, edges_path: str | Path):
    nodes_df = pd.read_csv(nodes_path)
    if "node_type" in nodes_df.columns:
        nodes_df = nodes_df[nodes_df["node_type"] == "ingredient"]

    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}

    for row in nodes_df.itertuples(index=False):
        norm = normalize_name(row.name)
        nid = int(row.node_id)
        name_to_id[norm] = nid
        id_to_name[nid] = norm

    edges_df = pd.read_csv(edges_path)
    if "edge_type" in edges_df.columns:
        edges_df = edges_df[edges_df["edge_type"] == "ingr-ingr"]

    pairing_lookup: dict[tuple[int, int], float] = {}
    for row in edges_df.itertuples(index=False):
        a, b, score = int(row.id_1), int(row.id_2), float(row.score)
        pairing_lookup[(a, b)] = score
        pairing_lookup[(b, a)] = score

    sorted_node_names = sorted(name_to_id.keys(), key=len, reverse=True)
    return name_to_id, id_to_name, pairing_lookup, sorted_node_names


# ---------------------------------------------------------
# 2. Clean Text & Extract Known FlavorGraph Ingredients
# ---------------------------------------------------------
UNIT_PATTERN = re.compile(
    r"\b\d*\s*[½¼¾⅓⅔⅛]?\s*"
    r"(cups?|tablespoons?|tbsp|teaspoons?|tsp|pounds?|lbs?|ounces?|oz|"
    r"grams?|g|kg|ml|cloves?|slices?|sheets?|cans?|packages?|pinch(es)?)\b",
    flags=re.IGNORECASE,
)


def lightweight_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = UNIT_PATTERN.sub(" ", text)
    text = re.sub(r"[\d½¼¾⅓⅔⅛/.-]+", " ", text)
    text = re.sub(r"[^a-z\s%]", " ", text)
    return " ".join(text.split()).strip()


def normalize_plurals(text: str) -> str:
    words = []
    for w in text.split():
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("oes"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def extract_nodes_from_text(
    raw_text: str,
    name_to_id: dict[str, int],
    sorted_node_names: list[str],
) -> set[int]:
    if not isinstance(raw_text, str):
        return set()

    cleaned = normalize_plurals(lightweight_clean(raw_text))
    padded = f" {cleaned} "
    found_ids: set[int] = set()

    for node_name in sorted_node_names:
        if len(node_name) < 3:
            continue
        target = f" {node_name} "
        if target in padded:
            found_ids.add(name_to_id[node_name])
            padded = padded.replace(target, " ")

    return found_ids


def ids_to_names(node_ids: set[int], id_to_name: dict[int, str]) -> list[str]:
    return sorted(id_to_name[nid] for nid in node_ids if nid in id_to_name)


# ---------------------------------------------------------
# 3. Prepare Recipes & Build ML Ranker
# ---------------------------------------------------------
def prepare_recipes(
    recipes_path: str | Path,
    name_to_id: dict[str, int],
    sorted_node_names: list[str],
):
    df = pd.read_csv(recipes_path, index_col=0)
    df = df.dropna(subset=["recipe_name", "ingredients", "directions"]).copy()
    df = df.drop_duplicates(subset=["recipe_name", "ingredients"]).reset_index(drop=True)

    if "rating" not in df.columns:
        df["rating"] = 0.0
    else:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)

    df["cleaned_text"] = df["ingredients"].apply(
        lambda raw: normalize_plurals(lightweight_clean(str(raw)))
    )
    df["node_ids"] = df["ingredients"].apply(
        lambda raw: extract_nodes_from_text(raw, name_to_id, sorted_node_names)
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_text"])
    return df, vectorizer, tfidf_matrix


def build_ml_ranker() -> LogisticRegression:
    """
    Train a scikit-learn classifier on feature rows:
    [main_cov, side_cov, avg_pairing, tfidf_score, main_hit]
    """
    X_train = np.array(
        [
            [1.0, 1.0, 0.25, 0.50, 1.0],
            [1.0, 0.5, 0.20, 0.40, 1.0],
            [0.5, 1.0, 0.20, 0.35, 1.0],
            [0.5, 0.5, 0.15, 0.30, 1.0],
            [0.5, 0.0, 0.10, 0.20, 1.0],
            [0.0, 1.0, 0.30, 0.40, 0.0],
            [0.0, 1.0, 0.15, 0.20, 0.0],
            [0.0, 0.5, 0.25, 0.25, 0.0],
            [0.0, 0.0, 0.20, 0.10, 0.0],
            [0.0, 0.0, 0.00, 0.00, 0.0],
        ]
    )
    y_train = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------
# 4. Flavor Pairing + Full Recipe Recommendation
# ---------------------------------------------------------
def recommend_full_recipe(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    model: LogisticRegression,
    main_ingredients: list[str],
    side_ingredients: list[str],
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
    sorted_node_names: list[str],
    pairing_lookup: dict[tuple[int, int], float],
    excluded_ids: set[int] | None = None,
    seen_indices: set[int] | None = None,
    top_n: int = 1,
):
    excluded_ids = excluded_ids or set()
    seen_indices = seen_indices or set()

    main_ids: set[int] = set()
    for ing in main_ingredients:
        main_ids |= extract_nodes_from_text(ing, name_to_id, sorted_node_names)

    side_ids: set[int] = set()
    for ing in side_ingredients:
        side_ids |= extract_nodes_from_text(ing, name_to_id, sorted_node_names)

    # Keep main ingredients distinct from side ingredients and remove excluded ones
    main_ids -= excluded_ids
    side_ids = (side_ids - main_ids) - excluded_ids
    user_ids = main_ids | side_ids

    # Suggest top FlavorGraph pairings for the user's available ingredients
    suggested_pairs: list[tuple[str, str, float]] = []
    for uid in sorted(user_ids):
        u_name = id_to_name.get(uid, str(uid))
        candidates = []
        for (a, b), score in pairing_lookup.items():
            if a == uid and b != uid and b not in excluded_ids:
                candidates.append((id_to_name.get(b, str(b)), score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        for partner_name, p_score in candidates[:3]:
            suggested_pairs.append((u_name, partner_name, round(p_score, 4)))

    query_terms = (
        [normalize_plurals(lightweight_clean(i)) for i in main_ingredients] * 3
        + [normalize_plurals(lightweight_clean(i)) for i in side_ingredients]
    )
    query_vec = vectorizer.transform([" ".join(query_terms)])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    feature_rows = []
    pairing_scores_list = []
    matched_names_list = []
    top_recipe_pairs_list = []
    main_hit_flags = []
    has_excluded_flags = []
    valid_indices = []

    for idx, recipe_nodes in enumerate(df["node_ids"]):
        if idx in seen_indices:
            continue

        matched_mains = main_ids & recipe_nodes
        matched_sides = side_ids & recipe_nodes
        matched_user = user_ids & recipe_nodes

        main_hit = 1.0 if len(matched_mains) > 0 else 0.0
        main_cov = (len(matched_mains) / len(main_ids)) if main_ids else 1.0
        side_cov = (len(matched_sides) / len(side_ids)) if side_ids else 1.0
        uses_excluded = bool(excluded_ids & recipe_nodes)

        other_nodes = recipe_nodes - user_ids
        pair_details = []
        for u in matched_user:
            for r in other_nodes:
                if r in excluded_ids:
                    continue
                s = pairing_lookup.get((u, r), 0.0)
                if s > 0:
                    pair_details.append((id_to_name.get(u, str(u)), id_to_name.get(r, str(r)), s))
        for a, b in combinations(sorted(matched_user), 2):
            s = pairing_lookup.get((a, b), 0.0)
            if s > 0:
                pair_details.append((id_to_name.get(a, str(a)), id_to_name.get(b, str(b)), s))

        pair_details.sort(key=lambda x: x[2], reverse=True)
        avg_pairing = float(np.mean([x[2] for x in pair_details])) if pair_details else 0.0

        feature_rows.append(
            [main_cov, side_cov, avg_pairing, float(tfidf_scores[idx]), main_hit]
        )
        pairing_scores_list.append(round(avg_pairing, 4))
        matched_names_list.append(ids_to_names(matched_user, id_to_name))
        top_recipe_pairs_list.append(pair_details[:5])
        main_hit_flags.append(main_hit)
        has_excluded_flags.append(uses_excluded)
        valid_indices.append(idx)

    if not valid_indices:
        return pd.DataFrame(), suggested_pairs

    X_candidates = np.array(feature_rows)
    ml_probs = model.predict_proba(X_candidates)[:, 1]

    adjusted_scores = []
    for prob, hit, excl in zip(ml_probs, main_hit_flags, has_excluded_flags):
        score = float(prob)
        if main_ids and hit == 0.0:
            score *= 0.1
        if excl:
            score *= 0.01  # Push recipes containing ingredients the user doesn't have to the back
        adjusted_scores.append(round(score, 4))

    ranked = df.loc[valid_indices].copy()
    ranked["ml_score"] = adjusted_scores
    ranked["flavor_pairing_score"] = pairing_scores_list
    ranked["matched_user_ingr"] = matched_names_list
    ranked["paired_flavors"] = top_recipe_pairs_list

    ranked = ranked.sort_values(
        by=["ml_score", "flavor_pairing_score", "rating"],
        ascending=[False, False, False],
    ).head(top_n)

    return ranked, suggested_pairs


def print_full_recipe(row: pd.Series, suggested_pairs: list[tuple[str, str, float]]):
    print("\n" + "=" * 60)
    print(f"RECIPE: {row['recipe_name']}")
    print(f"ML Match Score: {row['ml_score']} | Flavor Pairing Score: {row['flavor_pairing_score']}")
    print(f"Matched Ingredients You Have: {', '.join(row['matched_user_ingr']) or 'None'}")

    if row["paired_flavors"]:
        pairs_str = ", ".join(
            f"{a} + {b} ({score:.3f})" for a, b, score in row["paired_flavors"]
        )
        print(f"Paired Flavors in Recipe: {pairs_str}")
    elif suggested_pairs:
        pairs_str = ", ".join(
            f"{a} -> {b} ({score:.3f})" for a, b, score in suggested_pairs[:6]
        )
        print(f"Top FlavorGraph Pairings for Your Inputs: {pairs_str}")

    print("-" * 60)
    print("INGREDIENTS:")
    print(row["ingredients"])
    print("-" * 60)
    print("DIRECTIONS:")
    print(row["directions"])
    print("=" * 60 + "\n")


def parse_user_list(raw_input: str) -> list[str]:
    return [item.strip() for item in raw_input.split(",") if item.strip()]


# ---------------------------------------------------------
# 5. Interactive CLI Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    NODES_FILE = BASE_DIR / "external" / "inputs" / "nodes_191120.csv"
    EDGES_FILE = BASE_DIR / "external" / "inputs" / "edges_191120.csv"
    RECIPES_FILE = BASE_DIR / "recipes.csv"

    name_to_id, id_to_name, pairing_lookup, sorted_node_names = load_flavorgraph(
        NODES_FILE, EDGES_FILE
    )
    df, vectorizer, tfidf_matrix = prepare_recipes(
        RECIPES_FILE, name_to_id, sorted_node_names
    )
    ml_model = build_ml_ranker()

    main_raw = input("Enter your MAIN ingredients (comma-separated, e.g. apple, cranberry): ")
    side_raw = input("Enter your SIDE/PANTRY ingredients (comma-separated, e.g. butter, sugar): ")

    main_ingredients = parse_user_list(main_raw)
    main_set = {item.lower() for item in main_ingredients}
    side_ingredients = [
        item for item in parse_user_list(side_raw) if item.lower() not in main_set
    ]

    excluded_ids: set[int] = set()
    seen_indices: set[int] = set()

    while True:
        top_recipes, suggested_pairs = recommend_full_recipe(
            df=df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            model=ml_model,
            main_ingredients=main_ingredients,
            side_ingredients=side_ingredients,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
            sorted_node_names=sorted_node_names,
            pairing_lookup=pairing_lookup,
            excluded_ids=excluded_ids,
            seen_indices=seen_indices,
            top_n=1,
        )

        if top_recipes.empty:
            print("No more matching recipes left to try with your current filters.")
            break

        for idx, recipe_row in top_recipes.iterrows():
            seen_indices.add(int(idx))
            print_full_recipe(recipe_row, suggested_pairs)

        liked = input("Do you like this recipe and have the ingredients for it? (y/n): ").strip().lower()
        if liked in {"y", "yes"}:
            print("Awesome—enjoy making it!")
            break

        missing_raw = input(
            "Which ingredients from this recipe don't you have (comma-separated, or press Enter to skip)? "
        )
        for item in parse_user_list(missing_raw):
            excluded_ids |= extract_nodes_from_text(item, name_to_id, sorted_node_names)