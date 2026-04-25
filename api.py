"""
api.py — FastAPI server for the AutoStream AI assistant.

Improvements over v1:
  - Raw exception strings are never returned to clients.
  - Per-user rate limiting (max 30 requests/minute) to prevent abuse.
  - Session TTL cleanup (idle sessions removed after 2 hours).
  - Structured error responses with stable error codes.
  - Request body size validation.
  - /health endpoint for uptime monitoring.
"""

import time
import logging
from collections import defaultdict
from threading import Lock

from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

from agent import build_graph, run_turn, initial_state

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="AutoStream AI Assistant")
templates = Jinja2Templates(directory="templates")

graph = build_graph()

# ---------------------------------------------------------------------------
# Session store with TTL
# ---------------------------------------------------------------------------
SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours

sessions: dict = {}
session_timestamps: dict = {}
sessions_lock = Lock()


def _get_session(user_id: str) -> dict:
    with sessions_lock:
        now = time.time()
        # Expire idle sessions
        expired = [uid for uid, ts in session_timestamps.items() if now - ts > SESSION_TTL_SECONDS]
        for uid in expired:
            sessions.pop(uid, None)
            session_timestamps.pop(uid, None)
            logger.info(f"Session expired and removed: {uid}")

        state = sessions.get(user_id, initial_state())
        session_timestamps[user_id] = now
        return state


def _save_session(user_id: str, state: dict) -> None:
    with sessions_lock:
        sessions[user_id] = state
        session_timestamps[user_id] = time.time()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_WINDOW = 60      # seconds
RATE_LIMIT_MAX = 30         # requests per window

rate_counts: dict = defaultdict(list)
rate_lock = Lock()


def _is_rate_limited(user_id: str) -> bool:
    now = time.time()
    with rate_lock:
        timestamps = rate_counts[user_id]
        # Remove old timestamps outside window
        rate_counts[user_id] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        if len(rate_counts[user_id]) >= RATE_LIMIT_MAX:
            return True
        rate_counts[user_id].append(now)
        return False


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 1000


class ChatRequest(BaseModel):
    user_id: str
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"message exceeds {MAX_MESSAGE_LENGTH} character limit")
        return v

    @field_validator("user_id")
    @classmethod
    def user_id_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_id must not be empty")
        return v[:128]  # cap length


class ErrorResponse(BaseModel):
    error: str
    code: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.post("/chat")
def chat(req: ChatRequest):
    user_id = req.user_id

    # Rate limit check
    if _is_rate_limited(user_id):
        logger.warning(f"Rate limit hit for user: {user_id}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "Too many requests. Please slow down.", "code": "RATE_LIMITED"},
        )

    logger.info(f"[{user_id}] USER: {req.message[:120]}")

    try:
        state = _get_session(user_id)
        state, reply = run_turn(graph, state, req.message)
        _save_session(user_id, state)

        logger.info(f"[{user_id}] BOT: {reply[:120]}")

        return {
            "reply": reply,
            "intent": state.get("intent", "other"),
            "lead_stage": "captured" if state.get("lead_captured") else "in_progress",
        }

    except EnvironmentError as e:
        # Config error (missing API key etc.) — log details, return safe message
        logger.error(f"Configuration error for {user_id}: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "The assistant is temporarily unavailable due to a configuration issue. Please try again later.",
                "code": "CONFIG_ERROR",
            },
        )

    except Exception as e:
        logger.exception(f"Unhandled error for user {user_id}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Something went wrong on our end. Please try again in a moment.",
                "code": "INTERNAL_ERROR",
            },
        )


@app.get("/history")
def history(user_id: str):
    state = _get_session(user_id)
    if not state.get("messages"):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "No session found.", "code": "NOT_FOUND"},
        )

    clean_messages = [
        {
            "role": "user" if m.type == "human" else "assistant",
            "text": m.content,
        }
        for m in state["messages"]
        if m.content not in ("__INTERNAL_CAPTURE__", "__TRIGGER_LEAD_CAPTURE__", "__START_QUALIFICATION__")
    ]

    return {
        "messages": clean_messages,
        "intent": state.get("intent"),
        "lead_info": state.get("lead_info", {}),
        "lead_captured": state.get("lead_captured", False),
    }


@app.get("/debug")
def debug(user_id: str):
    state = _get_session(user_id)
    if not state.get("messages") and not state.get("intent"):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "No session found.", "code": "NOT_FOUND"},
        )
    return {
        "intent": state.get("intent"),
        "explained_topics": state.get("explained_topics", []),
        "lead_progress": state.get("lead_info", {}),
        "awaiting_field": state.get("awaiting_field"),
        "awaiting_signup_confirmation": state.get("awaiting_signup_confirmation"),
        "lead_captured": state.get("lead_captured"),
    }


@app.post("/reset")
def reset(user_id: str):
    with sessions_lock:
        sessions.pop(user_id, None)
        session_timestamps.pop(user_id, None)
    with rate_lock:
        rate_counts.pop(user_id, None)
    logger.info(f"Session reset: {user_id}")
    return {"message": "Session reset successful."}