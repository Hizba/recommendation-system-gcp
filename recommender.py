from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # si tu en as besoin ailleurs
from google.cloud import storage  # si tu l'utilises pour autre chose
import os


# ==============================
# 1) Variables globales
# ==============================

_ITEM_IDS = None
_ITEM_VECS = None
_DF_META = None
_DF_USER = None


# ==============================
# 2) Chargement des modèles
# ==============================

def _load_models_if_needed() -> None:
    """
    Charge en mémoire les modèles content-based :
      - _ITEM_IDS, _ITEM_VECS, _DF_META, _DF_USER depuis le dossier models/
    """
    global _ITEM_IDS, _ITEM_VECS, _DF_META, _DF_USER

    if _ITEM_IDS is not None and _ITEM_VECS is not None and _DF_META is not None and _DF_USER is not None:
        return

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

    user_rows = df_user[df_user["user_id"] == user_id]
    if user_rows.empty:
        return None
    user_row = user_rows.iloc[0]

    seen_ids = list(set(user_row["articles_list"]))
    if len(seen_ids) == 0:
        return None

    meta_user = df_meta[df_meta["article_id"].isin(seen_ids)][["article_id", "category_id"]]
    if meta_user.empty:
        return None

    cat_counts = meta_user["category_id"].value_counts()
    cat_weights = (cat_counts / cat_counts.sum()).to_dict()

    emb_df_tmp = pd.DataFrame(
        item_vecs,
        columns=[f"emb_{i}" for i in range(item_vecs.shape[1])]
    )
    emb_df_tmp.insert(0, "ar_ID", item_ids)

    emb_user = meta_user.merge(emb_df_tmp, left_on="article_id", right_on="ar_ID", how="inner")
    if emb_user.empty:
        return None

    weights = emb_user["category_id"].map(cat_weights).to_numpy().reshape(-1, 1)
    vecs = emb_user[[c for c in emb_df_tmp.columns if c != "ar_ID"]].to_numpy().astype(float)

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

    sims = cosine_similarity(profile.reshape(1, -1), item_vecs)[0]

    seen_ids = set(df_user[df_user["user_id"] == user_id]["articles_list"].iloc[0])
    mask_seen = np.isin(item_ids, list(seen_ids))
    sims[mask_seen] = -1.0

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
# 4) Interface pour main.py
# ==============================

def get_recommendations(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fonction appelée par main.py.

    input_data:
      - "user_id": int (obligatoire)
      - "k": int (optionnel, défaut=3)
      - "mode": ignoré ici, on fait toujours du content_based
    """
    user_id = input_data.get("user_id")
    if user_id is None:
        return []

    k = int(input_data.get("k", 5))

    recs_df = recommend_top_k_weighted(user_id=user_id, k=k)
    id_col = "ar_ID"

    if recs_df.empty:
        return []

    return [
        {"item_id": str(row[id_col]), "score": float(row["score"])}
        for _, row in recs_df.iterrows()
    ]
