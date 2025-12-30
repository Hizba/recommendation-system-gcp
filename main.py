import json
from typing import Any, Dict, Tuple

import functions_framework
from recommender import get_recommendations


def _error_response(message: str, status_code: int) -> Tuple[str, int, Dict[str, str]]:
    body = json.dumps({"error": message})
    headers = {"Content-Type": "application/json"}
    return body, status_code, headers


@functions_framework.http
def recommend(request):
    """
    HTTP Cloud Function pour les recommandations content-based.

    Body JSON attendu :
      {
        "user_id": 82705,   # obligatoire
        "k": 10             # optionnel (défaut 3)
      }
    """
    data = request.get_json(silent=True)
    if data is None:
        return _error_response("Requête JSON manquante ou invalide.", 400)

    user_id = data.get("user_id")
    if user_id is None:
        return _error_response("Champ 'user_id' requis dans le corps JSON.", 400)

    input_payload: Dict[str, Any] = {
        "user_id": user_id,
        "k": data.get("k", 5),
        # on force le mode content-based pour cette branche
        "mode": "content_based",
    }

    try:
        recommendations = get_recommendations(input_payload)

        response_body = json.dumps({"recommendations": recommendations})
        headers = {"Content-Type": "application/json"}
        return response_body, 200, headers

    except Exception as e:
        import traceback
        print(f"Error in recommend(): {e}")
        traceback.print_exc()
        return _error_response("Erreur interne du système de recommandation.", 500)
 