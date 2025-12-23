import json
from typing import Any, Dict, Tuple

import functions_framework
from werkzeug.wrappers import Response

from recommender import get_recommendations


def _error_response(message: str, status_code: int) -> Tuple[str, int, Dict[str, str]]:
    body = json.dumps({"error": message})
    headers = {"Content-Type": "application/json"}
    return body, status_code, headers


@functions_framework.http
def recommend(request) -> Response:
    """
    HTTP Cloud Run function for content recommendation.

    Expected JSON body:
      - {"user_id": 123}
      - or {"item_id": "ABC123"}

    Returns:
      JSON with "recommendations": list of { "item_id": str, "score": float }.
    """
    # Ensure JSON body
    data = request.get_json(silent=True)
    if data is None:
        return _error_response("Requête JSON manquante ou invalide.", 400)

    user_id = data.get("user_id")
    item_id = data.get("item_id")

    if user_id is None and item_id is None:
        return _error_response(
            "Champ 'user_id' ou 'item_id' requis dans le corps JSON.", 400
        )

    try:
        # Build an input dict that the recommender can understand
        input_payload: Dict[str, Any] = {}
        if user_id is not None:
            input_payload["user_id"] = user_id
        if item_id is not None:
            input_payload["item_id"] = item_id

        # Call business logic
        recommendations = get_recommendations(input_payload)

        response_body = json.dumps({"recommendations": recommendations})
        headers = {"Content-Type": "application/json"}
        return response_body, 200, headers

    except Exception as e:
        print("Error in recommend():", e)
        traceback.print_exc()
        return _error_response("Erreur interne du système de recommandation.", 500)
