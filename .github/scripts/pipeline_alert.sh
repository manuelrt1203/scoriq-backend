#!/usr/bin/env bash
# Signale dans une issue GitHub les étapes d'un job qui ont échoué, y compris celles marquées
# `continue-on-error: true` (leur `conclusion` reste « success », mais leur `outcome` vaut « failure »).
#
# Une issue ouverte par job (titre "[pipeline] <job> en échec", étiquette "pipeline") : un nouvel échec y ajoute
# un commentaire au lieu d'ouvrir un doublon ; un run réussi la ferme avec un commentaire « rétabli ».
# L'issue sert de journal, et GitHub prévient par e-mail.
#
# Variables d'environnement :
#   JOB_NAME          nom lisible du job surveillé
#   STEPS_JSON        toJSON(steps) du workflow (seules les étapes qui ont un `id` y figurent)
#   RUN_URL           lien vers le run
#   DETAILS           (facultatif) texte ajouté au message
#   GH_TOKEN          jeton GitHub (github.token), droit `issues: write`
set -euo pipefail

JOB_NAME="${JOB_NAME:?JOB_NAME manquant}"
RUN_URL="${RUN_URL:-}"
DETAILS="${DETAILS:-}"
STEPS_JSON="${STEPS_JSON:-}"
[ -n "$STEPS_JSON" ] || STEPS_JSON='{}'

LABEL="pipeline"
TITLE="[pipeline] ${JOB_NAME} en échec"
WHEN="$(date -u +'%Y-%m-%d %H:%M UTC')"

# Étapes en échec (liste d'identifiants séparés par des virgules).
FAILED="$(printf '%s' "$STEPS_JSON" | jq -r 'to_entries[] | select(.value.outcome == "failure") | .key' | paste -sd, - | sed 's/,/, /g')"

gh label create "$LABEL" --color D93F0B --description "Alerte du pipeline quotidien" --force >/dev/null 2>&1 || true
# Pas de `--search` : l'index de recherche de GitHub a un retard et créerait un doublon si deux échecs se suivent
# de près. On liste les issues ouvertes portant l'étiquette et on compare le titre exactement avec jq.
EXISTING="$(gh issue list --state open --label "$LABEL" --limit 100 --json number,title \
  | jq -r --arg t "$TITLE" '[.[] | select(.title == $t)][0].number // empty')"

if [ -n "$FAILED" ]; then
  BODY="**${WHEN}** — étape(s) en échec : \`${FAILED}\`"
  [ -n "$DETAILS" ] && BODY="${BODY}"$'\n\n'"${DETAILS}"
  [ -n "$RUN_URL" ] && BODY="${BODY}"$'\n\n'"Run : ${RUN_URL}"
  if [ -n "$EXISTING" ]; then
    gh issue comment "$EXISTING" --body "$BODY"
    echo "Échec signalé dans l'issue #${EXISTING}."
  else
    gh issue create --title "$TITLE" --label "$LABEL" --body "$BODY"
    echo "Issue d'alerte créée."
  fi
  exit 0
fi

# Un run annulé ou interrompu n'a rien validé : on ne ferme pas l'alerte sur cette base.
CANCELLED="$(printf '%s' "$STEPS_JSON" | jq -r 'to_entries[] | select(.value.outcome == "cancelled") | .key' | paste -sd, -)"
if [ -n "$CANCELLED" ]; then
  echo "Étape(s) annulée(s) (${CANCELLED}) : l'état de l'alerte est laissé tel quel."
  exit 0
fi

if [ -n "$EXISTING" ]; then
  gh issue comment "$EXISTING" --body "**${WHEN}** — rétabli : toutes les étapes ont réussi.${RUN_URL:+$'\n\n'Run : ${RUN_URL}}"
  gh issue close "$EXISTING"
  echo "Issue #${EXISTING} fermée (rétabli)."
else
  echo "Aucune étape en échec."
fi
