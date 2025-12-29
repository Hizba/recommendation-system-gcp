from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from surprise import AlgoBase
import joblib
from google.cloud import storage
import os


# ==============================
# 1) Variables globales
# ==============================

_ITEM_IDS = None
_ITEM_VECS = None
_DF_META = None
_DF_USER = None
_CF_MODEL = None        # modèle Surprise (SVD entraîné)
_CF_RATINGS_DF = None   # DataFrame ratings_df (user_id, article_id, rating)
# Config GCS pour le modèle collaboratif
_GCS_BUCKET_NAME = os.getenv("CF_MODELS_BUCKET", "projet-9-481708-models")
_GCS_SVD_PATH = os.getenv("CF_SVD_BLOB", "collab/svd_model.joblib")
_GCS_RATINGS_PATH = os.getenv("CF_RATINGS_BLOB", "collab/ratings_df.pkl")

_GCS_DOWNLOADED_SVD = "/tmp/svd_model.joblib"
_GCS_DOWNLOADED_RATINGS = "/tmp/ratings_df.pkl"


# ==============================
# 2) Chargement des modèles
# ==============================

def _load_models_if_needed() -> None:
    """
    Charge en mémoire :
      - modèle collaboratif (_CF_MODEL) + ratings_df (_CF_RATINGS_DF) depuis Cloud Storage
      - modèles content-based : _ITEM_IDS, _ITEM_VECS, _DF_META, _DF_USER depuis le dossier models/
    """
    global _ITEM_IDS, _ITEM_VECS, _DF_META, _DF_USER
    global _CF_MODEL, _CF_RATINGS_DF

    # 1) Modèle collaboratif depuis GCS
    if _CF_MODEL is None or _CF_RATINGS_DF is None:
        client = storage.Client()
        bucket = client.bucket(_GCS_BUCKET_NAME)

        print("GCS bucket:", _GCS_BUCKET_NAME)
        print("GCS SVD path:", _GCS_SVD_PATH)
        print("GCS ratings path:", _GCS_RATINGS_PATH)

        # Télécharger le modèle SVD
        blob_model = bucket.blob(_GCS_SVD_PATH)
        blob_model.download_to_filename(_GCS_DOWNLOADED_SVD)

        # Télécharger ratings_df
        blob_ratings = bucket.blob(_GCS_RATINGS_PATH)
        blob_ratings.download_to_filename(_GCS_DOWNLOADED_RATINGS)

        print("Local SVD path:", _GCS_DOWNLOADED_SVD)
        print("Local ratings path:", _GCS_DOWNLOADED_RATINGS)
        print("Local SVD size:", os.path.getsize(_GCS_DOWNLOADED_SVD))
        print("Local ratings size:", os.path.getsize(_GCS_DOWNLOADED_RATINGS))

        _CF_MODEL = joblib.load(_GCS_DOWNLOADED_SVD)
        _CF_RATINGS_DF = pd.read_pickle(_GCS_DOWNLOADED_RATINGS)

        print("Collaborative model (SVD) and ratings_df loaded from GCS.")

    # 2) Modèles content-based (toujours depuis le disque local du container)
    if _ITEM_IDS is None or _ITEM_VECS is None or _DF_META is None or _DF_USER is None:
        base_path = Path(__file__).parent / "models"
        emb_dir = base_path / "embeddings"
        meta_dir = base_path / "meta"
        user_dir = base_path / "user"

        _ITEM_IDS = np.load(emb_dir / "item_ids.npy", allow_pickle=True)
        _ITEM_VECS = np.load(emb_dir / "item_vecs.npy")

        _DF_META = pd.read_pickle(meta_dir / "df_meta.pkl")
        _DF_USER = pd.read_pickle(user_dir / "df_user.pkl")

        print("Content-based models loaded (embeddings + meta + user).")


# ==============================
# 3) Content-based pondéré
# ==============================

def build_user_profile_from_meta(user_id: int) -> np.ndarray | None:
    """Construit un profil utilisateur pondéré par catégories à partir de df_user, df_meta et des embeddings."""
    _load_models_if_needed()

    df_user = _DF_USER
    df_meta = _DF_META
    item_ids = _ITEM_IDS
    item_vecs = _ITEM_VECS

    # récupérer la ligne user
    user_rows = df_user[df_user["user_id"] == user_id]
    if user_rows.empty:
        return None
    user_row = user_rows.iloc[0]

    # articles vus par ce user (liste d'ar_ID)
    seen_ids = list(set(user_row["articles_list"]))
    if len(seen_ids) == 0:
        return None

    # sous-ensemble meta sur ces articles
    meta_user = df_meta[df_meta["article_id"].isin(seen_ids)][["article_id", "category_id"]]
    if meta_user.empty:
        return None

    # compter les catégories vues par ce user
    cat_counts = meta_user["category_id"].value_counts()
    cat_weights = (cat_counts / cat_counts.sum()).to_dict()

    # reconstruire un DataFrame embeddings pour la jointure
    emb_df_tmp = pd.DataFrame(
        item_vecs,
        columns=[f"emb_{i}" for i in range(item_vecs.shape[1])]
    )
    emb_df_tmp.insert(0, "ar_ID", item_ids)

    # jointure meta + embeddings
    emb_user = meta_user.merge(emb_df_tmp, left_on="article_id", right_on="ar_ID", how="inner")
    if emb_user.empty:
        return None

    # poids par article = poids de sa catégorie
    weights = emb_user["category_id"].map(cat_weights).to_numpy().reshape(-1, 1)
    vecs = emb_user[[c for c in emb_df_tmp.columns if c != "ar_ID"]].to_numpy().astype(float)

    # profil = moyenne pondérée
    profile = (vecs * weights).sum(axis=0) / weights.sum()
    return profile

def recommend_top_k_weighted(
    user_id: int,
    k: int = 3,
) -> pd.DataFrame:
    """
    Reco top-k content-based pondérée par catégories.
    Retourne un DataFrame avec colonnes ["ar_ID", "score"].
    """
    _load_models_if_needed()

    df_user = _DF_USER
    item_ids = _ITEM_IDS
    item_vecs = _ITEM_VECS

    profile = build_user_profile_from_meta(user_id)
    if profile is None:
        print("Impossible de construire un profil pour user:", user_id)
        return pd.DataFrame(columns=["ar_ID", "score"])

    # similarité profil vs tous les articles
    sims = cosine_similarity(profile.reshape(1, -1), item_vecs)[0]

    # exclure les articles déjà vus
    seen_ids = set(df_user[df_user["user_id"] == user_id]["articles_list"].iloc[0])
    mask_seen = np.isin(item_ids, list(seen_ids))
    sims[mask_seen] = -1.0

    # top-k
    k = min(k, len(sims))
    if k == 0:
        return pd.DataFrame(columns=["ar_ID", "score"])

    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return pd.DataFrame(
        {
            "ar_ID": item_ids[top_idx],
            "score": sims[top_idx],
        }
    )

# ==============================
# 4) Collaborative filtering
# ==============================

def recommend_top_k_collaborative(user_id: int, k: int = 5) -> pd.DataFrame:
    """
    Recommandations top-k avec le modèle collaboratif Surprise (SVD).
    Retourne un DataFrame ["article_id", "score"] pour ce user.
    """
    _load_models_if_needed()

    model = _CF_MODEL
    ratings_df = _CF_RATINGS_DF

    # Items déjà vus par ce user
    user_rows = ratings_df[ratings_df["user_id"] == user_id]
    if user_rows.empty:
        # user inconnu du modèle => on renvoie un DF vide (ou on bascule sur content-based)
        return pd.DataFrame(columns=["article_id", "score"])

    seen_items = set(user_rows["article_id"].unique())

    # Tous les items possibles
    all_items = ratings_df["article_id"].unique()

    # Candidats = items non vus
    candidates = [i for i in all_items if i not in seen_items]

    if not candidates:
        return pd.DataFrame(columns=["article_id", "score"])

    # Prédire les notes pour chaque item candidat
    preds = []
    for i in candidates:
        est = model.predict(str(user_id), str(i)).est
        preds.append((i, est))

    # Top-k
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)[:k]

    return pd.DataFrame(
        {
            "article_id": [i for i, _ in preds_sorted],
            "score": [est for _, est in preds_sorted],
        }
    )

# ==============================
# 5) Interface pour main.py
# ==============================

def get_recommendations(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fonction appelée par main.py.

    input_data:
      - "user_id": int (obligatoire)
      - "k": int (optionnel, défaut=3)
      - "mode": "content_based" ou "collab" (optionnel, défaut="content_based")
    """
    user_id = input_data.get("user_id")
    if user_id is None:
        return []

    k = int(input_data.get("k", 5))
    mode = input_data.get("mode", "content_based")

    if mode == "collab":
        # Collaborative filtering
        recs_df = recommend_top_k_collaborative(user_id=user_id, k=k)
        id_col = "article_id"
    else:
        # Content-based pondéré (mode par défaut)
        recs_df = recommend_top_k_weighted(user_id=user_id, k=k)
        id_col = "ar_ID"

    if recs_df.empty:
        return []

    return [
        {"item_id": str(row[id_col]), "score": float(row["score"])}
        for _, row in recs_df.iterrows()
    ]
