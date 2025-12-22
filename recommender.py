from typing import Any, Dict, List

import time

# TODO: importer ici numpy / pandas / sklearn si nécessaire.
# import numpy as np
# import pandas as pd
# from joblib import load

# Global model placeholder to ensure it is loaded once per instance.
_MODEL = None


def _load_model_if_needed() -> None:
    """
    Loads the recommendation model into the global _MODEL variable.

    This is called lazily on first request in a given Cloud Run instance.
    """
    global _MODEL
    if _MODEL is not None:
        return

    # TODO: remplacer ce bloc par un chargement réel du modèle.
    # Exemple avec un modèle local:
    #   from pathlib import Path
    #   model_path = Path(__file__).parent / "models" / "my_model.joblib"
    #   _MODEL = load(model_path)
    #
    # Exemple avec Cloud Storage:
    #   from google.cloud import storage
    #   client = storage.Client()
    #   bucket = client.bucket(MODEL_BUCKET)
    #   blob = bucket.blob(MODEL_BLOB_NAME)
    #   blob.download_to_filename("/tmp/my_model.joblib")
    #   _MODEL = load("/tmp/my_model.joblib")

    # Mock model for now
    print("Loading mock recommendation model...")
    time.sleep(0.1)  # simulate load time
    _MODEL = {"name": "mock_model", "version": "v1"}


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
