"""Garde-fous de l'API : validation du jeton Supabase, limitation de débit, taille des requêtes.

Pas de dépendance en plus : `requests` est déjà utilisé par le reste du backend.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque

import requests
from fastapi import HTTPException

# La clé « publishable » est publique par conception (elle est aussi dans le bundle
# du site) : elle identifie le projet, elle ne donne aucun droit. Les droits viennent
# du jeton de l'utilisateur, vérifié par Supabase.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://epmgjyiqcjklfjibrxhz.supabase.co").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_qGMZeILKhWfAC9dDQUn32Q_kwhaoGst")

TOKEN_CACHE_TTL = 60          # un jeton déjà vérifié (ou refusé) n'est pas revérifié pendant 60 s
TOKEN_CACHE_MAX = 2000
VALIDATIONS_PER_MINUTE = 120  # appels à Supabase pour des jetons jamais vus, tous visiteurs confondus
AGENT_REQUESTS = 30           # requêtes à l'assistant par utilisateur
AGENT_WINDOW = 600            # ... sur 10 minutes
AGENT_MAX_MESSAGES = 40
AGENT_MAX_CHARS = 50_000


class SlidingWindowLimiter:
    """Au plus `limit` événements par `window` secondes et par clé (mémoire du processus)."""

    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            q = self._events.setdefault(key, deque())
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self._events) > 10_000:
                self._purge(now)
            return True

    def _purge(self, now: float) -> None:
        for k in [k for k, q in self._events.items() if not q or q[-1] <= now - self.window]:
            del self._events[k]


_validation_budget = SlidingWindowLimiter(VALIDATIONS_PER_MINUTE, 60)
_agent_limiter = SlidingWindowLimiter(AGENT_REQUESTS, AGENT_WINDOW)
_token_cache: dict[str, tuple[float, str | None]] = {}
_token_lock = threading.Lock()


def _fetch_user_id(token: str) -> str | None:
    """Demande à Supabase à qui appartient ce jeton. None = refusé ; exception = service injoignable."""
    r = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_PUBLISHABLE_KEY},
        timeout=5,
    )
    if r.status_code in (401, 403):
        return None
    r.raise_for_status()
    uid = r.json().get("id")
    return uid if isinstance(uid, str) and uid else None


def authenticate_bearer(authorization: str | None) -> str:
    """Renvoie l'identifiant Supabase de l'utilisateur, ou lève 401 / 429 / 503."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Connexion requise pour utiliser l'assistant.")
    token = authorization[7:].strip()
    if not token or len(token) > 4096:
        raise HTTPException(status_code=401, detail="Connexion requise pour utiliser l'assistant.")

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
            if len(_token_cache) >= TOKEN_CACHE_MAX:
                _token_cache.clear()
        _token_cache[digest] = (now + TOKEN_CACHE_TTL, uid)
    if uid is None:
        raise HTTPException(status_code=401, detail="Session invalide ou expirée, reconnecte-toi.")
    return uid


def check_agent_quota(user_id: str) -> None:
    if not _agent_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Tu as beaucoup sollicité l'assistant, réessaie dans quelques minutes.")


def check_agent_payload(messages: list[dict]) -> None:
    """Borne la taille de la conversation envoyée au modèle (chaque message est facturé en jetons)."""
    if len(messages) > AGENT_MAX_MESSAGES:
        raise HTTPException(status_code=413, detail="Conversation trop longue, recommence une nouvelle discussion.")
    if len(json.dumps(messages, default=str)) > AGENT_MAX_CHARS:
        raise HTTPException(status_code=413, detail="Conversation trop volumineuse, recommence une nouvelle discussion.")
