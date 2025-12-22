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
    Core business logic for computing recommendations.

    Args:
        input_data: dict containing "user_id" and/or "item_id".

    Returns:
        A list of dicts: [{"item_id": str, "score": float}, ...]
    """
    _load_model_if_needed()

    user_id = input_data.get("user_id")
    item_id = input_data.get("item_id")

    # Simple mock logic: generate deterministic dummy results
    base_id = f"user-{user_id}" if user_id is not None else f"item-{item_id}"

    recommendations = [
        {"item_id": f"{base_id}-rec-1", "score": 0.92},
        {"item_id": f"{base_id}-rec-2", "score": 0.87},
        {"item_id": f"{base_id}-rec-3", "score": 0.81},
    ]

    return recommendations
