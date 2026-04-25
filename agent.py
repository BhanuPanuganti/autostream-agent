"""
agent.py — AutoStream Conversational AI Agent built with LangGraph.

Improvements over v1:
  - Hallucination prevention: LLM is ONLY used when KB context is sufficient;
    all plan/policy facts use hardcoded canonical responses.
  - Sentinel string leak prevention: __TRIGGER_LEAD_CAPTURE__ and
    __START_QUALIFICATION__ are never exposed to the user.
  - Fixed: "video" keyword hijacking unrelated flows.
  - Fixed: intent always falling back to "other" in LLM path.
  - Fixed: await_recommendation state interfering with normal FAQ answers.
  - Cleaner error messages — no raw exception strings returned to the client.
"""

import os
import json
import re
from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from rag import retrieve, get_full_kb_summary
from tools import mock_lead_capture, classify_intent_heuristic, Intent
from rag import _load_kb

KB = _load_kb()


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class LeadInfo(TypedDict, total=False):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    lead_info: LeadInfo
    lead_captured: bool
    awaiting_field: Optional[str]
    awaiting_signup_confirmation: bool
    explained_topics: list
    awaiting_recommendation: bool
    awaiting_decision_help: bool


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    model_name = os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash-lite")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is not set.\n"
            "Export it with: export GOOGLE_API_KEY=your_key_here"
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,   # lower = less hallucination
    )


# ---------------------------------------------------------------------------
# Canonical hardcoded responses — SINGLE SOURCE OF TRUTH
# No LLM touches these facts.
# ---------------------------------------------------------------------------
def _build_plans_response() -> str:
    plans = KB.get("plans", [])

    if len(plans) < 2:
        return "Plan information is currently unavailable."

    basic = next((p for p in plans if "basic" in p["name"].lower()), None)
    pro = next((p for p in plans if "pro" in p["name"].lower()), None)

    if not basic or not pro:
        return "Plan information is incomplete."
    
    # Format the feature lists as markdown bullets instead of a brittle table
    basic_features = "\n".join([f"- {feat}" for feat in basic["features"]])
    pro_features = "\n".join([f"- {feat}" for feat in pro["features"]])

    return f"""**AutoStream Plans Comparison**

**Basic Plan (${basic['price_monthly']}/mo)**
{basic_features}

**Pro Plan (${pro['price_monthly']}/mo)**
{pro_features}

• **Basic** — best for lighter workflows.
• **Pro** — best for scaling creators who need unlimited output and priority support.

Would you like help choosing the right plan for your workflow?"""

def _get_policy(topic: str) -> str:
    for p in KB.get("policies", []):
        if topic.lower() in p["topic"].lower():
            return f"**{p['topic']}**\n\n{p['detail']}"
    return "Policy not found."


def _get_faq(keyword: str) -> str:
    """Dynamically pull FAQ answers from the JSON knowledge base."""
    for faq in KB.get("faqs", []):
        if keyword.lower() in faq["question"].lower() or keyword.lower() in faq["answer"].lower():
            return f"**{faq['question']}**\n\n{faq['answer']}"
    return "I don't have that information handy."

def _build_features_response() -> str:
    """Build a clear, product-oriented feature list from JSON."""
    plans = KB.get("plans", [])

    basic = next((p for p in plans if "basic" in p["name"].lower()), None)
    pro = next((p for p in plans if "pro" in p["name"].lower()), None)

    if not basic or not pro:
        return "Feature information is currently unavailable."

    # Format lists directly without trying to force set intersections
    basic_features = "\n".join(f"- {f}" for f in basic["features"])
    pro_features = "\n".join(f"- {f}" for f in pro["features"])

    return f"""**AutoStream Plans Comparison**

**Basic Plan (${basic['price_monthly']}/mo)**
{basic_features}

**Pro Plan (${pro['price_monthly']}/mo)**
{pro_features}

• **Basic** — best for lighter workflows.
• **Pro** — best for scaling creators who need unlimited output and priority support.

Which plan feels like the better fit for you?""" # <--- THE FIX

OUT_OF_SCOPE = (
    "I can help specifically with AutoStream plans, pricing, features, policies, and signup. "
    "Could you rephrase your question or ask about one of those topics?"
)

def _get_all_policies() -> str:
    """Dynamically build a summary of all policies."""
    policies = KB.get("policies", [])
    if not policies:
        return "Policy information is currently unavailable."
    
    # Friendly, conversational header
    response = "Here's a quick overview of how we handle things:\n\n"
    for p in policies:
        response += f"**{p['topic']}**: {p['detail']}\n\n"
    
    response += "Does that answer your questions, or can I help with anything else?"
    return response.strip()


MENU_RESPONSE = (
    "I can help you with:\n\n"
    "- Plans & pricing (Basic $29/mo, Pro $79/mo)\n"
    "- Features & capabilities\n"
    "- Refund, billing & cancellation policies\n"
    "- Free trial info\n"
    "- Getting started / signup\n\n"
    "What would you like to know?"
)


# ---------------------------------------------------------------------------
# Keyword matching helpers
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PLATFORM_KEYWORDS = [
    "youtube", "instagram", "tiktok", "twitter", "x", "facebook",
    "twitch", "linkedin", "snapchat", "pinterest", "vimeo",
]

PLAN_RECOMMEND_PATTERNS = re.compile(
    r"\b(which|what)\s+(plan|tier)\b|\bbest\s+plan\b|\brecommend(?:ed|ation)?\b|"
    r"\bshould i (pick|choose|get)\b|\bwhich one\b",
    re.IGNORECASE,
)
PRICE_OBJECTION_PATTERNS = re.compile(
    r"\b(expensive|costly|too much|overpriced|over budget|budget|afford|cheap)\b",
    re.IGNORECASE,
)
PLAN_DISSATISFACTION_PATTERNS = re.compile(
    r"\b(don.?t|do not|not)\b.*\b(like|happy|satisfied)\b.*\b(plan|subscription)\b|"
    r"\b(plan|subscription)\b.*\b(not working|isn.?t working|not useful|broken|issue|problem)\b|"
    r"\bpromised\b.*\b(not|isn.?t|wasn.?t)\b.*\b(given|delivered|provided)\b",
    re.IGNORECASE,
)
REFUND_PATTERNS = re.compile(
    r"\b(refund|money back|return my money|cancel and refund)\b",
    re.IGNORECASE,
)
PRO_SIGNALS = [
    "unlimited", "4k", "captions", "caption", "priority", "advanced",
    "branding", "daily", "many videos", "high volume", "team", "agency",
    "long video", "youtube", "tiktok", "instagram", "grow", "scale",
]
BASIC_SIGNALS = [
    "budget", "cheap", "affordable", "light usage", "few videos",
    "starter", "beginner", "simple", "small",
]


def _keyword_score(text: str, keywords: list) -> int:
    lower = text.lower()
    return sum(1 for kw in keywords if kw in lower)


def _infer_plan(user_text: str) -> Optional[str]:
    pro = _keyword_score(user_text, PRO_SIGNALS)
    basic = _keyword_score(user_text, BASIC_SIGNALS)
    if "trial" in user_text.lower():
        pro += 1
    if pro > basic:
        return "pro"
    if basic > pro:
        return "basic"
    return None


def _is_yes(text: str) -> bool:
    return text.strip().lower() in {
        "yes", "y", "yeah", "yep", "sure", "ok", "okay",
        "sounds good", "let's do it", "lets do it", "alright",
    }


def _is_no(text: str) -> bool:
    return text.strip().lower() in {
        "no", "n", "nope", "not now", "later", "maybe later", "not yet",
    }


def _is_explicit_start_now(text: str) -> bool:
    lower = text.strip().lower()
    return lower in {
        "start", "start now", "yes start", "lets start", "let's start",
        "sign me up", "get started", "proceed", "begin",
    }


def _is_closure_ack(text: str) -> bool:
    return text.strip().lower() in {
        "thanks", "thank you", "thankyou", "thx", "appreciate it", "ty",
    }


def _next_missing_field(lead_info: dict) -> Optional[str]:
    for field in ["name", "email", "platform"]:
        if not lead_info.get(field):
            return field
    return None


def _clean_name_value(text: str) -> Optional[str]:
    value = text.strip()
    value = re.sub(r"^(hi|hello|hey)[,\s]+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(i am|i'm|im|my name is|this is|it's|its)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[.!?]+$", "", value).strip()
    if not value or any(t in value.lower() for t in ("@", "http", "www.")):
        return None
    if len(value.split()) > 5:
        return None
    value = re.sub(r"[^A-Za-z\s\-']", "", value).strip()
    if not value:
        return None
    return " ".join(p.capitalize() for p in value.split())


def _extract_field(field: str, user_text: str) -> Optional[str]:
    text = user_text.strip()
    if field == "email":
        m = EMAIL_RE.search(text)
        return m.group(0) if m else None
    if field == "platform":
        lower = text.lower()
        for kw in PLATFORM_KEYWORDS:
            if kw in lower:
                return kw.capitalize()
        if len(text.split()) <= 4:
            return text.strip()
        return None
    if field == "name":
        return _clean_name_value(text)
    return None


def _extract_topics(user_text: str) -> list:
    lower = user_text.lower()
    topics = []
    if "basic" in lower:
        topics.append("basic_plan")
    if "pro" in lower and "product" not in lower:
        topics.append("pro_plan")
    if any(w in lower for w in ("price", "pricing", "cost", "plan", "plans")):
        topics.append("pricing")
    if "refund" in lower:
        topics.append("refund_policy")
    if "bill" in lower:
        topics.append("billing")
    if "trial" in lower:
        topics.append("free_trial")
    if "cancel" in lower:
        topics.append("cancellation")
    if "feature" in lower:
        topics.append("features")
    return list(dict.fromkeys(topics))


# ---------------------------------------------------------------------------
# Plan recommendation responses — still deterministic
# ---------------------------------------------------------------------------

def _plan_recommendation_response(user_text: str) -> str:
    plan = _infer_plan(user_text)
    if plan == "pro":
        return (
            "Based on what you've shared, **Pro** is the better fit.\n\n"
            "It includes unlimited videos, 4K exports (up to 4 hours), AI-generated captions, "
            "advanced scene detection, custom branding watermarks, and priority 24/7 support.\n\n"
            "You can also start with the **7-day free trial** — no credit card required.\n\n"
            "Want to get started now?"
        )
    if plan == "basic":
        return (
            "Based on your needs, **Basic** is a smart starting point at $29/month.\n\n"
            "You get 10 videos/month, 720p exports, basic auto-cut and trim, and standard email support.\n\n"
            "You can upgrade to Pro at any time — changes apply at the next billing cycle.\n\n"
            "Would you like to start with Basic, or see a full side-by-side comparison first?"
        )
    return (
        "Here's a quick way to choose:\n\n"
        "- **Basic ($29/month)** — up to 10 videos/month, 720p exports, standard support.\n"
        "- **Pro ($79/month)** — unlimited videos, 4K exports, AI captions, advanced scene detection, "
        "custom branding, 24/7 priority support, and a free 7-day trial.\n\n"
        "If you're aiming to grow your output, Pro typically gives more headroom. "
        "How many videos do you publish per month?"
    )


def _price_objection_response(user_text: str) -> str:
    plan = _infer_plan(user_text)
    if plan == "basic":
        return (
            "Fair point on budget. **Basic at $29/month** covers core editing for up to 10 videos — "
            "you can always upgrade to Pro later when your publishing volume grows.\n\n"
            "Want to go with Basic for now?"
        )
    return (
        "Totally fair. **Pro at $79/month** is designed to replace multiple separate tools "
        "(captioning, editing, branding, scheduling) — so for higher-volume creators it often saves cost overall.\n\n"
        "There's also a **7-day free trial with no credit card required**, so you can test it before committing.\n\n"
        "Want to try Pro risk-free?"
    )


def _plan_dissatisfaction_response() -> str:
    return (
        "Sorry to hear that — let's sort it out quickly.\n\n"
        "Here are your options:\n\n"
        "1) **Switch plans** — You can upgrade or downgrade anytime; the change applies at the next billing cycle.\n"
        "2) **Get support** — Pro includes priority 24/7 live support; Basic includes standard email support (24h response).\n"
        "3) **Request a refund** — If your purchase is within the first 7 days, you're eligible for a refund.\n\n"
        "Which would be most helpful right now?"
    )


def _plan_switch_response(target_plan: str) -> str:
    if target_plan == "basic":
        return (
            "**Basic Plan — $29/month**\n\n"
            "- 10 videos/month, up to 30 min each\n"
            "- 720p exports\n"
            "- Basic auto-cut and trim\n"
            "- Standard email support (24h response)\n\n"
            "Compared to Pro, it's more budget-friendly but has usage and feature limits. "
            "Want a full side-by-side comparison?"
        )
    return (
        "**Pro Plan — $79/month**\n\n"
        "- Unlimited videos, up to 4 hours each\n"
        "- 4K exports\n"
        "- AI-generated captions\n"
        "- Advanced scene detection\n"
        "- Custom branding watermarks\n"
        "- Priority 24/7 live support\n"
        "- 7-day free trial (no card required)\n\n"
        "Compared to Basic, it's built for higher-volume and higher-quality workflows. "
        "Want to start the free trial?"
    )


# ---------------------------------------------------------------------------
# Deterministic KB lookup — no LLM needed for these
# ---------------------------------------------------------------------------

def _deterministic_response(user_text: str, last_ai_lower: str = "") -> Optional[str]:
    lower = user_text.lower()

    # --------------------------------------------------
    # 1. RECOMMENDATION (HIGHEST PRIORITY)
    # --------------------------------------------------
    if _is_plan_recommendation_query(user_text):
        return _plan_recommendation_response(user_text)

    # --------------------------------------------------
    # 2. PRICE OBJECTION
    # --------------------------------------------------
    if _is_price_objection(user_text):
        return _price_objection_response(user_text)

    # --------------------------------------------------
    # 3. PLAN DISSATISFACTION
    # --------------------------------------------------
    if _is_plan_dissatisfaction(user_text):
        return _plan_dissatisfaction_response()

    # --------------------------------------------------
    # 4. SPECIFIC PLAN (basic / pro)
    # --------------------------------------------------
    if "basic" in lower and not any(w in lower for w in ("compare", "vs", "difference", "recommend", "which", "better")):
        return _plan_switch_response("basic")

    if "pro" in lower and not any(w in lower for w in ("compare", "vs", "difference", "recommend", "which", "better", "product")):
        return _plan_switch_response("pro")

    # --------------------------------------------------
    # 5. POLICY HANDLER
    # --------------------------------------------------
    if "refund" in lower:
        return _get_policy("refund")

    if "cancel" in lower:
        return _get_policy("cancel")

    if "billing" in lower or "bill" in lower:
        return _get_policy("billing")

    if "policy" in lower or "policies" in lower:
        return (
            "I can explain:\n\n"
            "• Refund policy\n"
            "• Cancellation policy\n"
            "• Billing details\n\n"
            "What would you like to know?"
        )
    
    #--------------------------------------------------
    # 6. OTHER DETERMINISTIC QUERIES (FAQs & Features)
    # --------------------------------------------------
    if "trial" in lower or "free trial" in lower:
        return _get_policy("trial") # "Free Trial" is under policies in your JSON

    if ("platform" in lower or "publish" in lower or "where" in lower) and ("post" in lower or "upload" in lower):
        return _get_faq("platform")

    if ("video" in lower and "limit" in lower) or ("long" in lower and "video" in lower) or ("video length" in lower) or ("max" in lower and "video" in lower):
        return _get_faq("limit")

    if "support" in lower and not any(x in lower for x in ["refund", "cancel", "billing"]):
        return _get_policy("support")

    if "feature" in lower and not any(w in lower for w in ("which", "best", "recommend", "compare")):
        return _build_features_response()

    # --------------------------------------------------
    # 7. GENERIC PLAN QUERY (LOW PRIORITY)
    # --------------------------------------------------
    if _is_generic_plan_query(lower):
        return _build_plans_response()

    # --------------------------------------------------
    # 8. FALLBACK
    # --------------------------------------------------
    return None

def _is_generic_plan_query(lower: str) -> bool:
    asks_plan = "plans" in lower or lower.strip() in {"plan", "pricing", "price"}
    asks_pricing = "pricing" in lower or "price" in lower or "cost" in lower
    mentions_specific = "basic" in lower or "pro" in lower
    
    # NEW: Added 'compare' and 'detail' to trigger the full list
    is_question = any(w in lower for w in ["what", "show", "list", "compare", "detail"])

    # Trigger if it's a generic question OR explicitly asks to compare, as long as it doesn't mention a specific plan yet
    return (asks_plan or asks_pricing or is_question or "compare" in lower) and not mentions_specific


def _is_plan_recommendation_query(text: str) -> bool:
    return bool(PLAN_RECOMMEND_PATTERNS.search(text))


def _is_price_objection(text: str) -> bool:
    return bool(PRICE_OBJECTION_PATTERNS.search(text))


def _is_plan_dissatisfaction(text: str) -> bool:
    return bool(PLAN_DISSATISFACTION_PATTERNS.search(text))


def _is_refund_query(text: str) -> bool:
    return bool(REFUND_PATTERNS.search(text))


# ---------------------------------------------------------------------------
# LLM-grounded generation (fallback for queries not covered deterministically)
# ---------------------------------------------------------------------------

GROUNDED_PROMPT = """You are Aria, AutoStream's customer support assistant.

RULES — follow every one strictly:
1. Answer ONLY using information in the <knowledge_base> below.
2. If the answer is not in the knowledge base, say exactly: "I don't have that information — please contact our support team."
3. Never invent plans, prices, features, policies, discounts, integrations, or guarantees.
4. Keep your answer concise (2–4 sentences max). Be friendly, not corporate.
5. End with a short follow-up question or CTA where natural.
6. Do NOT repeat greetings. Do NOT say "Of course!" or "Great question!".

<knowledge_base>
{kb_context}
</knowledge_base>

User: {user_query}
Aria:"""


def _llm_grounded_generate(user_query: str, llm_instance) -> str:
    """Generate a response strictly grounded in KB context. LLM = formatter, not fact-source."""
    kb_context = _get_kb_context(user_query)
    if not kb_context or "no relevant" in kb_context.lower():
        return OUT_OF_SCOPE

    prompt = GROUNDED_PROMPT.format(kb_context=kb_context, user_query=user_query)
    try:
        response = llm_instance.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip() or OUT_OF_SCOPE
    except Exception as err:
        return _friendly_llm_fallback(err)


def _friendly_llm_fallback(err: Exception) -> str:
    lower = str(err).lower()
    if "resource_exhausted" in lower or "quota" in lower or "429" in lower:
        return (
            "I'm at capacity for a moment. Quick facts: Basic is $29/month, "
            "Pro is $79/month with a 7-day free trial. Ask me anything about plans, "
            "features, or policies!"
        )
    if "not_found" in lower or "404" in lower:
        return "I'm experiencing a configuration issue. Please try again shortly."
    if "api key" in lower or "permission_denied" in lower or "unauthorized" in lower:
        return "There's an authentication issue on my end. Please try again later."
    return "I had a momentary issue — could you rephrase your question?"


def _strip_redundant_greeting(user_text: str, ai_text: str) -> str:
    if classify_intent_heuristic(user_text) == Intent.GREETING:
        return ai_text
    return re.sub(r"^(hi|hello|hey there|hi there)[!.\s,]*", "", ai_text, flags=re.IGNORECASE).strip()


def _get_kb_context(query: str) -> str:
    try:
        return retrieve(query)
    except Exception:
        return get_full_kb_summary()


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

llm = None  # lazy init


def _ensure_llm():
    global llm
    if llm is None:
        llm = _get_llm()
    return llm


def node_respond(state: AgentState) -> AgentState:
    """
    Core response node. Priority order:
    1. Sentinel/internal state handling (field collection, confirmations)
    2. Deterministic KB responses (no LLM needed → zero hallucination risk)
    3. LLM-grounded fallback (for genuinely ambiguous queries)
    """
    messages = state["messages"]
    lead_info = state.get("lead_info", {})
    awaiting = state.get("awaiting_field")
    explained_topics = state.get("explained_topics", [])
    awaiting_signup_confirmation = state.get("awaiting_signup_confirmation", False)
    awaiting_recommendation = state.get("awaiting_recommendation", False)
    awaiting_decision_help = state.get("awaiting_decision_help", False)

    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    last_ai = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage)), ""
    )
    user_text = last_human.strip()
    user_lower = user_text.lower()
    last_ai_lower = last_ai.lower()

    # ------------------------------------------------------------------
    # 1. Pure greeting
    # ------------------------------------------------------------------
    if user_lower in {"hi", "hello", "hey", "hii", "helo", "hi there", "hey there"}:
        return _reply(state, "Hi! I'm Aria, AutoStream's assistant.\n\nI can help you explore plans, features, policies, or get you started with signup. What would you like to know?", intent="greeting")

    # ------------------------------------------------------------------
    # 2. Post-Capture Exit & Closure
    # ------------------------------------------------------------------
    if state.get("lead_captured"):
        if _is_closure_ack(user_text):
            name = (lead_info.get("name") or "").strip()
            msg = f"You're welcome{', ' + name if name else ''}! If you need anything in the future, just reach out."
            return _reply(state, msg, intent="other")
            
        # NEW: Catch "no", "nothing", "bye" specifically after capture
        if _is_no(user_text) or user_lower in {"nothing", "none", "bye", "goodbye", "exit", "quit"}:
            name = (lead_info.get("name") or "").strip()
            msg = f"Sounds good{', ' + name if name else ''}! We'll be in touch soon. Have a great day!"
            return _reply(state, msg, intent="other")

    # ------------------------------------------------------------------
    # 3. Signup confirmation gate
    # ------------------------------------------------------------------
    if awaiting_signup_confirmation:
        if _is_yes(user_text) or _is_explicit_start_now(user_text):
            return _start_qualification(state)
        if _is_no(user_text):
            return _reply(
                state,
                "No problem! I can help you compare plans, pricing, and policies. What would you like to know?",
                intent="product_inquiry",
                awaiting_signup_confirmation=False,
            )
        return _reply(
            state,
            "Would you like to start signup now? Please reply **yes** or **no**.",
            intent="other",
            awaiting_signup_confirmation=True,
        )

    # ------------------------------------------------------------------
    # 4. Explicit start now (no confirmation needed)
    # ------------------------------------------------------------------
    if not awaiting and _is_explicit_start_now(user_text):
        return _start_qualification(state)

    # ------------------------------------------------------------------
    # 5. Field collection mode
    # ------------------------------------------------------------------
    if awaiting:
        # Allow product questions during field collection — answer, then re-prompt
        intent_hint = classify_intent_heuristic(user_text)
        if intent_hint == Intent.PRODUCT_INQUIRY:
            det = _deterministic_response(user_text, last_ai_lower)
            if det:
                reply = det
            else:
                reply = _llm_grounded_generate(user_text, _ensure_llm())
            field_prompt = {
                "name": "When you're ready, could you share your name?",
                "email": "When you're ready, could you share your email address?",
                "platform": "And which platform do you mainly create on?",
            }.get(awaiting, "When you're ready, we can continue with signup.")
            return _reply(state, f"{reply}\n\n{field_prompt}", intent="product_inquiry", awaiting_field=awaiting)

        extracted = _extract_field(awaiting, user_text)
        if extracted:
            lead_info = {**lead_info, awaiting: extracted}
            next_field = _next_missing_field(lead_info)
            if next_field:
                prompts = {
                    "name": "Got it! Could you share your name?",
                    "email": f"Thanks{', ' + lead_info.get('name', '') if lead_info.get('name') else ''}! What's your email address?",
                    "platform": "Almost there! Which platform do you mainly create content on? (YouTube, Instagram, TikTok, etc.)",
                }
                return _reply(state, prompts[next_field], intent="high_intent",
                              awaiting_field=next_field, lead_info=lead_info)
            else:
                # All fields collected — route instantly using state flag
                return {
                    **state,
                    "lead_info": lead_info,
                    "awaiting_field": None,
                    "intent": "high_intent",
                    "_trigger_capture": True,
                }
        else:
            retries = {
                "email": "That doesn't look like a valid email. Please use format: name@example.com",
                "platform": "Which platform do you mainly create on? (e.g., YouTube, Instagram, TikTok)",
                "name": "Could you share your name as you'd like to be addressed?",
            }
            return _reply(state, retries.get(awaiting, "Could you share that again?"),
                          intent="high_intent", awaiting_field=awaiting, lead_info=lead_info)

    # ------------------------------------------------------------------
    # 6. Awaiting video count recommendation
    # ------------------------------------------------------------------
    if awaiting_recommendation:
        # Only trigger volume-based recommendation if user provides a number
        number_match = re.search(r"\b(\d+)\b", user_text)
        if number_match:
            count = int(number_match.group(1))
            if count <= 10:
                suggestion = "Based on that volume, **Basic ($29/month)** should cover your current workflow — 10 videos/month at 720p."
            else:
                suggestion = f"With {count} videos/month, **Pro ($79/month)** is the better fit — unlimited videos, 4K exports, and AI captions."
            return _reply(
                state,
                f"{suggestion}\n\nWould you like to get started, or compare plans in detail?",
                intent="product_inquiry",
                awaiting_recommendation=False,
            )
        # Non-numeric reply — ask again clearly
        if _is_yes(user_text) or _is_no(user_text):
            return _reply(
                state,
                "Roughly how many videos do you publish per month? A number like 5 or 20 works great.",
                intent="product_inquiry",
                awaiting_recommendation=True,
            )
        # They may have answered something else — check deterministic first
        det = _deterministic_response(user_text)
        if det:
            return _reply(state, det, intent="product_inquiry")

    # ------------------------------------------------------------------
    # 7. Decision help menu
    # ------------------------------------------------------------------
    if awaiting_decision_help:
        if user_lower == "1":
            return _reply(state, _build_features_response(), intent="product_inquiry",
                          awaiting_decision_help=False,
                          explained_topics=list(dict.fromkeys(explained_topics + ["features"])))
        if user_lower == "2":
            return _reply(state, _build_plans_response(), intent="product_inquiry",
                          awaiting_decision_help=False,
                          explained_topics=list(dict.fromkeys(explained_topics + ["pricing"])))

    # ------------------------------------------------------------------
    # 8. "Not sure" / uncertainty flows
    # ------------------------------------------------------------------

    if any(phrase in user_lower for phrase in ["not sure", "dont know", "don't know", "idk", "unsure"]):
        return _reply(
            state,
            "No worries! Let me help you choose.\n\nRoughly how many videos do you publish per month?",
            intent="product_inquiry",
            awaiting_recommendation=True,
        )

    # ------------------------------------------------------------------
    # 9. Yes/No follow-up on prior AI message
    # ------------------------------------------------------------------
    # NEW: Catch "all" requests specifically for the policies menu
    if user_lower in {"all", "all of them", "explain all", "explain all of them"} and any(w in last_ai_lower for w in ["refund", "cancellation", "billing", "policy"]):
        return _reply(state, _get_all_policies(), intent="product_inquiry")
    
    if _is_yes(user_text) and not awaiting_signup_confirmation:
        # NEW: Catch "yes" to "Would you like help choosing?"
        if "help choosing" in last_ai_lower or "right plan" in last_ai_lower:
            return _reply(
                state,
                "Awesome! To point you in the right direction, roughly how many videos do you publish per month?",
                intent="product_inquiry",
                awaiting_recommendation=True
            )

        # "yes" after a CTA to get started
        if any(phrase in last_ai_lower for phrase in ["get started", "start now", "want to start", "ready to start", "try pro", "start your free trial", "start signup"]):
            return _start_qualification(state)
            
        # "yes" after a plan comparison CTA
        if any(phrase in last_ai_lower for phrase in ["compare", "side-by-side"]):
            return _reply(state, _build_plans_response(), intent="product_inquiry",
                          explained_topics=list(dict.fromkeys(explained_topics + ["pricing"])))
                          
        if "pro" in user_lower or "basic" in user_lower:
            return _start_qualification(state)
            
        # Generic yes — smooth out the tone here too (removing the overly casual emoji)
        return _reply(
            state,
            "Got it. What would you like to explore next?\n\n"
            "1) Compare plans\n"
            "2) See features\n"
            "3) Learn policies\n"
            "4) Start signup",
            intent="product_inquiry"
        )

    if _is_no(user_text) and not awaiting_signup_confirmation:
        return _reply(state, "No problem. What else can I help you with?", intent="product_inquiry")

    # ------------------------------------------------------------------
    # 11. Deterministic KB lookup (covers 95% of real queries, no LLM)
    # ------------------------------------------------------------------
    lower = user_text.lower()
    
    det = _deterministic_response(user_text, last_ai_lower)
    if det:
        new_topics = _extract_topics(user_text)
        
        # Check if the bot is asking the recommendation question
        is_asking_volume = "How many videos do you publish per month?" in det
        
        return _reply(
            state, 
            det, 
            intent="product_inquiry",
            explained_topics=list(dict.fromkeys(explained_topics + new_topics)),
            awaiting_recommendation=is_asking_volume  # Dynamically set the flag
        )
    
    # 🔥 SMART AUTO-RECOMMENDATION
    number_match = re.search(r"\b(\d+)\b", user_text)
    if number_match and any(w in user_lower for w in ["video", "videos", "month"]):
        count = int(number_match.group(1))

        if count <= 10:
            return _reply(
                state,
                "Since you're creating around {} videos/month, *Basic ($29/month)* should be enough for now.\n\n"
                "You can always upgrade later. Want help getting started?".format(count),
                intent="product_inquiry"
            )
        else:
            return _reply(
                state,
                "With around {} videos/month, *Pro ($79/month)* is a better fit — it gives you unlimited videos, 4K exports, and AI captions.\n\n"
                "Want to start the free trial?".format(count),
                intent="product_inquiry"
            )

    # ------------------------------------------------------------------
    # 12. Repeated topic short-circuit
    # ------------------------------------------------------------------
    current_topics = _extract_topics(user_text)
    repeated = [t for t in current_topics if t in explained_topics]
    if repeated and len(current_topics) == 1 and not any(w in user_lower for w in ("again", "recap", "remind", "more", "tell me")):
        topic_name = repeated[0].replace("_", " ")
        return _reply(state, f"We've covered {topic_name} — want a quick recap or shall we look at something else?", intent="product_inquiry")

    # ------------------------------------------------------------------
    # 13. Out-of-scope: truly unknown intent
    # ------------------------------------------------------------------
    if classify_intent_heuristic(user_text) == Intent.OTHER and not current_topics:
        return _reply(state, MENU_RESPONSE, intent="other")

    # ------------------------------------------------------------------
    # 14. LLM fallback — only reached for ambiguous but KB-answerable queries
    # ------------------------------------------------------------------
    generated = _llm_grounded_generate(user_text, _ensure_llm())
    generated = _strip_redundant_greeting(user_text, generated)
    new_topics = _extract_topics(user_text)
    return _reply(state, generated, intent="product_inquiry",
                  explained_topics=list(dict.fromkeys(explained_topics + new_topics)))


# ---------------------------------------------------------------------------
# Helper: build state update dict cleanly
# ---------------------------------------------------------------------------

def _reply(
    state: AgentState,
    text: str,
    intent: str = "product_inquiry",
    awaiting_field: Optional[str] = None,
    lead_info: Optional[dict] = None,
    awaiting_signup_confirmation: bool = False,
    awaiting_recommendation: bool = False,
    awaiting_decision_help: bool = False,
    explained_topics: Optional[list] = None,
) -> AgentState:
    """Build a state update with a clean AI message. No sentinel strings."""
    update = {
        **state,
        "messages": [AIMessage(content=text)],
        "intent": intent,
        "awaiting_field": awaiting_field,
        "awaiting_signup_confirmation": awaiting_signup_confirmation,
        "awaiting_recommendation": awaiting_recommendation,
        "awaiting_decision_help": awaiting_decision_help,
    }
    if lead_info is not None:
        update["lead_info"] = lead_info
    if explained_topics is not None:
        update["explained_topics"] = explained_topics
    return update


def _start_qualification(state: AgentState) -> AgentState:
    """Transition into lead qualification — ask for the first missing field."""
    lead_info = state.get("lead_info", {})
    next_field = _next_missing_field(lead_info)
    if not next_field:
        # Already have all fields — go straight to capture
        return {
            **state,
            "messages": [AIMessage(content="__INTERNAL_CAPTURE__")],
            "intent": "high_intent",
            "_trigger_capture": True,
        }
    prompts = {
        "name": "Let's get you set up. Could I start with your name?",
        "email": f"Thanks{', ' + lead_info.get('name', '') if lead_info.get('name') else ''}! What's your email address?",
        "platform": "Which platform do you mainly create content on? (YouTube, Instagram, TikTok, etc.)",
    }
    return _reply(state, prompts[next_field], intent="high_intent",
                  awaiting_field=next_field, awaiting_signup_confirmation=False,
                  lead_info=lead_info)


# ---------------------------------------------------------------------------
# Remaining nodes
# ---------------------------------------------------------------------------

def node_qualify_lead(state: AgentState) -> AgentState:
    lead_info = state.get("lead_info", {})
    next_field = _next_missing_field(lead_info)
    if not next_field:
        return {**state, "awaiting_field": None}
    prompts = {
        "name": "Let's get you set up. Could I start with your name?",
        "email": f"Thanks{', ' + lead_info.get('name', '') if lead_info.get('name') else ''}! What's your email address?",
        "platform": "Which platform do you mainly create content on? (YouTube, Instagram, TikTok, etc.)",
    }
    return _reply(state, prompts[next_field], intent="high_intent", awaiting_field=next_field)


def node_capture_lead(state: AgentState) -> AgentState:
    if state.get("lead_captured"):
        return state  # idempotency
    lead = state.get("lead_info", {})
    try:
        mock_lead_capture(
            name=lead.get("name", "Unknown"),
            email=lead.get("email", "Unknown"),
            platform=lead.get("platform", "Unknown"),
        )
    except Exception:
        pass  # mock — never fail the user flow

    name = lead.get("name", "")
    platform = lead.get("platform", "")
    confirmation = (
        f"🎉 You're all set{', ' + name if name else ''}! "
        f"We've recorded your details{' for ' + platform if platform else ''} and our team will be in touch shortly.\n\n"
        "Welcome to AutoStream! Is there anything else I can help you with?"
    )
    return {
        **state,
        "messages": [AIMessage(content=confirmation)],
        "lead_captured": True,
        "awaiting_field": None,
        "intent": "high_intent",
    }


def node_confirm_signup(state: AgentState) -> AgentState:
    return _reply(
        state,
        "Happy to help you get started! Would you like to begin signup now? (yes / no)",
        intent="other",
        awaiting_signup_confirmation=True,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_respond(state: AgentState) -> str:
    if state.get("lead_captured"):
        return END

    # Check for internal capture trigger (stored in state, NOT in message text)
    if state.get("_trigger_capture"):
        return "capture_lead"

    # Check last AI message for internal sentinel (guard: never shown to user)
    last_ai = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), ""
    )
    if last_ai == "__INTERNAL_CAPTURE__":
        return "capture_lead"

    intent = state.get("intent", "other")
    awaiting = state.get("awaiting_field")
    awaiting_signup = state.get("awaiting_signup_confirmation", False)

    if intent == "high_intent" and not awaiting:
        lead_info = state.get("lead_info", {})
        if not awaiting_signup and _next_missing_field(lead_info) is not None:
            return "confirm_signup"
        if _next_missing_field(lead_info) is None:
            return "capture_lead"
        return "qualify_lead"

    return END


def route_after_qualify(state: AgentState) -> str:
    if _next_missing_field(state.get("lead_info", {})) is None:
        return "capture_lead"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("respond", node_respond)
    graph.add_node("confirm_signup", node_confirm_signup)
    graph.add_node("qualify_lead", node_qualify_lead)
    graph.add_node("capture_lead", node_capture_lead)
    graph.set_entry_point("respond")
    graph.add_conditional_edges(
        "respond",
        route_after_respond,
        {
            "confirm_signup": "confirm_signup",
            "qualify_lead": "qualify_lead",
            "capture_lead": "capture_lead",
            END: END,
        },
    )
    graph.add_edge("confirm_signup", END)
    graph.add_conditional_edges(
        "qualify_lead",
        route_after_qualify,
        {"capture_lead": "capture_lead", END: END},
    )
    graph.add_edge("capture_lead", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_turn(graph, state: AgentState, user_input: str) -> tuple:
    new_state = graph.invoke({
        **state,
        "messages": state["messages"] + [HumanMessage(content=user_input)],
    })
    # Ensure sentinel messages are never returned to callers
    reply = next(
        (m.content for m in reversed(new_state["messages"]) if isinstance(m, AIMessage)),
        "",
    )
    if reply in ("__INTERNAL_CAPTURE__", "__TRIGGER_LEAD_CAPTURE__", "__START_QUALIFICATION__"):
        reply = "Just a moment — processing your request..."
    return new_state, reply


def initial_state() -> AgentState:
    return {
        "messages": [],
        "intent": None,
        "lead_info": {},
        "lead_captured": False,
        "awaiting_field": None,
        "awaiting_signup_confirmation": False,
        "explained_topics": [],
        "awaiting_recommendation": False,
        "awaiting_decision_help": False,
        "_trigger_capture": False,
    }