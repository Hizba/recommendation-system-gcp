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
    HTTP Cloud Run function pour les recommandations.

    Body JSON attendu :
      {
        "user_id": 82705,      # obligatoire
        "k": 10,               # optionnel (défaut 3)
        "mode": "content_based" ou "collab"  # optionnel (défaut "content_based")
      }
    """
    # Lire le JSON
    data = request.get_json(silent=True)
    if data is None:
        return _error_response("Requête JSON manquante ou invalide.", 400)

    user_id = data.get("user_id")
    if user_id is None:
        return _error_response("Champ 'user_id' requis dans le corps JSON.", 400)

    # Construire le payload pour la couche reco
    input_payload: Dict[str, Any] = {
        "user_id": user_id,
        "k": data.get("k", 5),
        "mode": data.get("mode", "content_based"),
    }

    try:
        # Appel de la logique métier
        recommendations = get_recommendations(input_payload)

        response_body = json.dumps({"recommendations": recommendations})
        headers = {"Content-Type": "application/json"}
        return response_body, 200, headers

    except Exception as e:
        # Log minimal visible dans Cloud Logging
        import traceback
        print(f"Error in recommend(): {e}")
        traceback.print_exc()

        return _error_response("Erreur interne du système de recommandation.", 500)
