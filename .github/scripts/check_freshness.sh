#!/usr/bin/env bash
# Vérifie que l'API publique répond et que les données sont fraîches. Échoue (code 1) sinon, avec la
# raison dans $GITHUB_OUTPUT (clé « reason ») pour l'alerte.
#
# L'API tourne sur l'offre gratuite de Render : elle s'endort et met jusqu'à ~1 min à se réveiller, d'où
# les tentatives répétées. Seuil de fraîcheur : dernier match ÉVALUÉ de moins de MAX_AGE_DAYS jours (7 par
# défaut : assez large pour les trêves internationales, assez serré pour déceler un pipeline arrêté).
set -uo pipefail

API="${API_URL:-https://scoriq-backend.onrender.com}"
# Trêve estivale (juin à août) : aucune compétition suivie ne joue pendant des semaines, un seuil de 7 jours
# déclencherait une fausse alerte quotidienne et finirait par être ignoré.
case "$(date -u +%m)" in
  06|07|08) DEFAULT_MAX_AGE=35 ;;
  *)        DEFAULT_MAX_AGE=7 ;;
esac
MAX_AGE_DAYS="${MAX_AGE_DAYS:-$DEFAULT_MAX_AGE}"
OUT="${GITHUB_OUTPUT:-/dev/null}"

fail() {
  echo "ÉCHEC : $1"
  # La raison ne contient que du texte que nous écrivons nous-mêmes (pas de contenu de l'API recopié tel quel).
  echo "reason=$1" >> "$OUT"
  exit 1
}

fetch() {
  curl --fail --silent --show-error --max-time 90 --retry "${CURL_RETRY:-4}" --retry-delay "${CURL_RETRY_DELAY:-20}" \
    --retry-all-errors "$API$1"
}

summary="$(fetch /stats/summary)" || fail "L'API ne répond pas (/stats/summary) après plusieurs tentatives."
evaluated="$(printf '%s' "$summary" | jq -r '.evaluated_predictions // empty' 2>/dev/null)"
case "$evaluated" in
  ''|*[!0-9]*) fail "/stats/summary ne renvoie pas de nombre de prédictions évaluées." ;;
esac

latest="$(fetch '/predictions/history?limit=1&only_evaluated=true' | jq -r '.[0].match_date // empty' 2>/dev/null)" \
  || fail "L'API ne répond pas (/predictions/history)."
case "$latest" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
  *) fail "Aucune prédiction évaluée récente n'a pu être lue." ;;
esac

now_s="$(date -u +%s)"
latest_s="$(date -u -d "$latest" +%s 2>/dev/null)" || fail "Date du dernier match évalué illisible."
age_days=$(( (now_s - latest_s) / 86400 ))
echo "API OK : ${evaluated} prédictions évaluées, dernier match évalué le ${latest} (il y a ${age_days} j, seuil ${MAX_AGE_DAYS} j)."
if [ "$age_days" -gt "$MAX_AGE_DAYS" ]; then
  fail "Données périmées : dernier match évalué le ${latest} (il y a ${age_days} jours, seuil ${MAX_AGE_DAYS})."
fi
