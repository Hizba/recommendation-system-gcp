from typing import Any, Dict, List

import time

# TODO: importer ici numpy / pandas / sklearn si nécessaire.
# import numpy as np
# import pandas as pd
# from joblib import load

# Global model placeholder to ensure it is loaded once per instance.
_MODEL = None


def _load_models_if_needed() -> None:
    global _ITEM_IDS, _ITEM_VECS, _DF_META, _DF_USER

    if _ITEM_IDS is not None:
        return

    base_path = Path(__file__).parent / "models"
    emb_dir = base_path / "embeddings"
    meta_dir = base_path / "meta"
    user_dir = base_path / "user"

    _ITEM_IDS = np.load(emb_dir / "item_ids.npy", allow_pickle=True)
    _ITEM_VECS = np.load(emb_dir / "item_vecs.npy")

    _DF_META = pd.read_pickle(meta_dir / "df_meta.pkl")
    _DF_USER = pd.read_pickle(user_dir / "df_user.pkl")

    print("Models loaded in memory (embeddings + meta + user).")


def get_recommendations(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Utilise recommend_top_k_weighted pour renvoyer les vrais IDs d'articles (ar_ID).
    """
    user_id = input_data.get("user_id")
    if user_id is None:
        return []

    k = int(input_data.get("k", 3))

    # DF avec colonnes ["ar_ID", "score"]
    recs_df = recommend_top_k_weighted(user_id=user_id, df_user=_DF_USER, df_meta=_DF_META, emb_df=None, k=k)

    # On renvoie les IDs d'articles tels quels
    return [
        {"item_id": str(row["ar_ID"]), "score": float(row["score"])}
        for _, row in recs_df.iterrows()
    ]
