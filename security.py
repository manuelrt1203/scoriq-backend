"""Garde-fous de l'API : validation du jeton Supabase, limitation de débit, taille des requêtes.

Pas de dépendance en plus : `requests` est déjà utilisé par le reste du backend.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
from collections import deque

import requests
from fastapi import HTTPException

logger = logging.getLogger("scoriq")

# La clé « publishable » est publique par conception (elle est aussi dans le bundle
# du site) : elle identifie le projet, elle ne donne aucun droit. Les droits viennent
# du jeton de l'utilisateur, vérifié par Supabase.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://epmgjyiqcjklfjibrxhz.supabase.co").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_qGMZeILKhWfAC9dDQUn32Q_kwhaoGst")

TOKEN_CACHE_TTL = 60          # un jeton déjà vérifié (ou refusé) n'est pas revérifié pendant 60 s
TOKEN_CACHE_MAX = 2000
VALIDATIONS_PER_MINUTE = 300  # appels à Supabase pour des jetons jamais vus, tous visiteurs confondus
# Une requête peut déclencher jusqu'à AGENT_MAX_TOOL_ROUNDS appels à Groq : les plafonds restent bas.
AGENT_REQUESTS = 40           # requêtes à l'assistant par utilisateur (une question avec outil client en coûte 2)
AGENT_WINDOW = 600            # ... sur 10 minutes
AGENT_GLOBAL_REQUESTS = 100   # plafond tous utilisateurs confondus (comptes créés en série)
AGENT_MAX_MESSAGES = 60
AGENT_MAX_CHARS = 250_000     # historique complet, résultats d'outils renvoyés par le client compris
AGENT_ROLES = {"user", "assistant"}  # jamais « system » ni « tool » : ils viennent du serveur


class SlidingWindowLimiter:
    """Au plus `limit` événements par `window` secondes et par clé (mémoire du processus)."""

    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_purge = float("-inf")

    def would_allow(self, key: str) -> bool:
        """Comme allow(), sans enregistrer l'événement."""
        now = self._clock()
        with self._lock:
            q = self._events.get(key)
            if q is None:
                return self.limit > 0
            while q and q[0] <= now - self.window:
                q.popleft()
            return len(q) < self.limit

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            q = self._events.setdefault(key, deque())
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            # Balayage des clés inactives, au plus une fois par dixième de fenêtre (sinon chaque appel
            # parcourrait 10 000 clés sous le verrou).
            if len(self._events) > 10_000 and now - self._last_purge > self.window / 10:
                self._last_purge = now
                self._purge(now)
            return True

    def _purge(self, now: float) -> None:
        for k in [k for k, q in self._events.items() if not q or q[-1] <= now - self.window]:
            del self._events[k]


_validation_budget = SlidingWindowLimiter(VALIDATIONS_PER_MINUTE, 60)
_agent_limiter = SlidingWindowLimiter(AGENT_REQUESTS, AGENT_WINDOW)
_agent_global_limiter = SlidingWindowLimiter(AGENT_GLOBAL_REQUESTS, AGENT_WINDOW)
_token_cache: dict[str, tuple[float, str | None]] = {}
_token_lock = threading.Lock()


def _looks_like_live_jwt(token: str, now: float | None = None) -> bool:
    """Filtre local, sans réseau : trois segments, charge utile JSON avec `sub` et `exp` non dépassé.
    Écarte les chaînes aléatoires sans dépenser le budget d'appels à Supabase (la vraie
    vérification de la signature reste faite par Supabase)."""
    parts = token.split(".")
    if len(parts) != 3 or not all(parts) or len(parts[1]) > 2048:
        return False
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, TypeError, RecursionError):  # JSON profondément imbriqué : RecursionError
        return False
    if not isinstance(claims, dict) or not isinstance(claims.get("sub"), str):
        return False
    exp = claims.get("exp")
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        return False
    return exp > (time.time() if now is None else now)


def _is_api_key_rejection(r) -> bool:
    """Le rejet de la clé API vient de la passerelle de Supabase : JSON {"message": ...} SANS les champs
    de l'API d'authentification (error_code / msg). Un jeton forgé dont l'en-tête contiendrait « api key »
    (repris dans le message d'erreur d'authentification) ne peut donc pas déclencher cette alerte."""
    try:
        body = r.json()
    except ValueError:
        return False
    if not isinstance(body, dict) or "error_code" in body or "msg" in body:
        return False
    message = str(body.get("message", "")).lower()
    return "api key" in message or "apikey" in message


def _fetch_user_id(token: str) -> str | None:
    """Demande à Supabase à qui appartient ce jeton. None = refusé ; exception = service injoignable."""
    r = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_PUBLISHABLE_KEY},
        timeout=5,
    )
    detail = (getattr(r, "text", "") or "")[:200]
    if r.status_code in (401, 403):
        # 401/403 = jeton refusé... sauf si c'est NOTRE clé API qui est refusée (mauvaise configuration) :
        # on le dit dans les journaux et on répond 503, au lieu de déconnecter tout le monde en silence.
        if _is_api_key_rejection(r):
            logger.error("Supabase refuse la clé publishable du backend : %s", detail)
            raise requests.RequestException("clé API refusée")
        return None
    if r.status_code != 200:
        # Autre 4xx (mauvaise URL, requête mal formée), 429, 5xx : ce n'est pas un verdict sur le jeton.
        logger.error("Réponse inattendue de Supabase (%s) : %s", r.status_code, detail)
        raise requests.RequestException(f"statut {r.status_code}")
    body = r.json()
    if not isinstance(body, dict) or body.get("is_anonymous") is True:
        return None  # les connexions anonymes permettraient de contourner le quota par compte
    uid = body.get("id")
    return uid if isinstance(uid, str) and uid else None


def authenticate_bearer(authorization: str | None) -> str:
    """Renvoie l'identifiant Supabase de l'utilisateur, ou lève 401 / 429 / 503."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Connexion requise pour utiliser l'assistant.")
    token = authorization[7:].strip()
    if not token or len(token) > 4096 or not _looks_like_live_jwt(token):
        raise HTTPException(status_code=401, detail="Session invalide ou expirée, reconnecte-toi.")

    digest = hashlib.sha256(token.encode()).hexdigest()  # le jeton lui-même n'est jamais gardé
    now = time.monotonic()
    with _token_lock:
        hit = _token_cache.get(digest)
    if hit and hit[0] > now:
        if hit[1] is None:
            raise HTTPException(status_code=401, detail="Session invalide ou expirée, reconnecte-toi.")
        return hit[1]

    if not _validation_budget.allow("global"):
        raise HTTPException(status_code=429, detail="Trop de tentatives, réessaie dans une minute.")
    try:
        uid = _fetch_user_id(token)
    except (requests.RequestException, ValueError):
        # Échec fermé : si on ne peut pas vérifier, on ne laisse pas passer.
        raise HTTPException(status_code=503, detail="Vérification de la session indisponible, réessaie.")

    with _token_lock:
        if len(_token_cache) >= TOKEN_CACHE_MAX:
            for k in [k for k, (exp, _) in _token_cache.items() if exp <= now]:
                del _token_cache[k]
        # Si le cache est encore plein d'entrées valides on ne mémorise pas celle-ci : on ne vide
        # jamais tout (cela ferait revérifier d'un coup tous les utilisateurs légitimes).
        if len(_token_cache) < TOKEN_CACHE_MAX:
            _token_cache[digest] = (now + TOKEN_CACHE_TTL, uid)
    if uid is None:
        raise HTTPException(status_code=401, detail="Session invalide ou expirée, reconnecte-toi.")
    return uid


def check_agent_quota(user_id: str) -> None:
    # Deux temps : on vérifie les deux plafonds SANS rien enregistrer, puis on enregistre. Une requête refusée
    # ne consomme donc ni le plafond global (un seul compte insistant ne bloque pas les autres) ni le quota
    # personnel (un refus dû à la saturation globale ne pénalise pas l'utilisateur).
    if not _agent_limiter.would_allow(user_id):
        raise HTTPException(status_code=429, detail="Tu as beaucoup sollicité l'assistant, réessaie dans quelques minutes.")
    if not _agent_global_limiter.would_allow("global"):
        raise HTTPException(status_code=429, detail="L'assistant est très sollicité, réessaie dans quelques minutes.")
    _agent_limiter.allow(user_id)
    _agent_global_limiter.allow("global")


def _valid_block(b) -> bool:
    """Forme minimale des blocs lus par main._wire_messages_to_groq_messages : sans elle, un bloc
    incomplet lèverait KeyError/AttributeError (500 sans en-têtes CORS). Types inconnus : ignorés plus loin."""
    if not isinstance(b, dict):
        return False
    t = b.get("type")
    if t == "text":
        return isinstance(b.get("text"), str)
    if t == "tool_use":
        return isinstance(b.get("id"), str) and isinstance(b.get("name"), str)
    if t == "tool_result":
        return isinstance(b.get("tool_use_id"), str) and isinstance(b.get("content"), (str, type(None)))
    return True


def check_agent_payload(messages: list[dict]) -> None:
    """Borne la conversation envoyée au modèle (chaque message est facturé en jetons) et refuse
    les rôles que seul le serveur peut écrire, pour qu'un client ne puisse pas injecter un
    message « system »."""
    if len(messages) > AGENT_MAX_MESSAGES:
        raise HTTPException(status_code=413, detail="Conversation trop longue, recommence une nouvelle discussion.")
    for m in messages:
        if (not isinstance(m, dict) or not isinstance(m.get("role"), str) or m["role"] not in AGENT_ROLES
                or not isinstance(m.get("content"), (str, list))):
            raise HTTPException(status_code=400, detail="Message invalide.")
        if isinstance(m["content"], list) and not all(_valid_block(b) for b in m["content"]):
            raise HTTPException(status_code=400, detail="Message invalide.")
    if len(json.dumps(messages, default=str, ensure_ascii=False)) > AGENT_MAX_CHARS:
        raise HTTPException(status_code=413, detail="Conversation trop volumineuse, recommence une nouvelle discussion.")
