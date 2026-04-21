"""
agent.py — AutoStream Conversational AI Agent built with LangGraph.

Architecture:
  - LangGraph StateGraph manages the conversation flow across nodes
  - State carries: messages, intent, lead_info (name/email/platform), collected flags
  - RAG context is retrieved and injected per-turn
  - Tool (mock_lead_capture) is called only when all 3 lead fields are collected
"""

import os
from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import retrieve, get_full_kb_summary
from tools import mock_lead_capture, classify_intent_heuristic, Intent

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class LeadInfo(TypedDict, total=False):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    intent: str                                # last classified intent
    lead_info: LeadInfo                        # collected lead fields
    lead_captured: bool                        # whether tool was already called
    awaiting_field: Optional[str]              # which field we're currently asking for
    awaiting_signup_confirmation: bool          # waiting for yes/no before qualification
    explained_topics: list[str]                # tracks topics already explained


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

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are Aria, a helpful and friendly AI sales assistant for AutoStream — a SaaS platform offering automated video editing tools for content creators.

Your goals:
1. Greet users warmly and understand their needs.
2. Answer product, pricing, and policy questions accurately using ONLY the knowledge base below.
3. Identify when a user shows high buying intent (e.g. wants to sign up, try the Pro plan, get started).
4. When high intent is detected, collect the user's name, email, and creator platform (YouTube, Instagram, TikTok, etc.) — one at a time, naturally.
5. NEVER call the lead capture tool until you have all three: name, email, and platform.

Intent classification rules (include in your response as JSON on the LAST line ONLY):
- "greeting" → user is saying hi / small talk
- "product_inquiry" → user asking about features, pricing, policies
- "high_intent" → user clearly wants to sign up now / start now / proceed now
- "other" → anything else

Important intent nuance:
- If the user is still exploring, comparing, considering, or undecided (even if they mention trying later), classify as "product_inquiry".
- Only classify as "high_intent" when the user explicitly indicates they are ready to proceed now.

Conversation quality rules:
- Avoid repeating the same details if they were already explained in this conversation.
- If a user asks about an already covered topic, give a short summary and move the conversation forward.
- Do not greet repeatedly. Only greet when the user greets first.

Accuracy rules:
- Never invent plans, prices, or policy details.
- AutoStream has only two plans in this knowledge base: Basic Plan ($29/month) and Pro Plan ($79/month).
- If asked about plans, answer using only those two plans.

IMPORTANT:
Return ONLY valid JSON on the last line in this exact format:
{{"intent": "<intent_value>"}}
Do not add any text after the JSON.

Knowledge Base:
{kb_context}

Current lead info collected so far: {lead_info}
"""

def _build_system_prompt(kb_context: str, lead_info: dict) -> str:
    lead_str = str({k: v for k, v in lead_info.items() if v}) or "nothing yet"
    return SYSTEM_PROMPT_TEMPLATE.format(
        kb_context=kb_context,
        lead_info=lead_str
    )


# ---------------------------------------------------------------------------
# Helper: parse intent from LLM response
# ---------------------------------------------------------------------------

import json, re

def _parse_intent_from_response(text: str) -> tuple[str, str]:
    """
    Extract the intent JSON tag from the LLM response.
    Returns (clean_text, intent_str).
    """
    pattern = r'\{"intent":\s*"([^"]+)"\}'
    match = re.search(pattern, text)
    if match:
        intent = match.group(1)
        clean = text[:match.start()].strip()
        return clean, intent
    return text.strip(), "other"


# ---------------------------------------------------------------------------
# Helper: extract field value from a short user reply
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PLATFORM_KEYWORDS = ["youtube", "instagram", "tiktok", "twitter", "x", "facebook",
                     "twitch", "linkedin", "snapchat", "pinterest", "vimeo"]

def _extract_field(field: str, user_text: str) -> Optional[str]:
    """Try to extract a specific lead field from a short user reply."""
    text = user_text.strip()
    if field == "email":
        m = EMAIL_RE.search(text)
        return m.group(0) if m else None
    if field == "platform":
        lower = text.lower()
        for kw in PLATFORM_KEYWORDS:
            if kw in lower:
                return kw.capitalize()
        # Accept short free-text answers as platform name
        if len(text.split()) <= 4:
            return text
        return None
    if field == "name":
        # Accept anything up to 5 words that isn't a URL/email
        if len(text.split()) <= 5 and "@" not in text and "http" not in text:
            return text
        return None
    return None


def _extract_topics(user_text: str) -> list[str]:
    """Detect high-level topics mentioned by the user."""
    lower = user_text.lower()
    topics = []
    if "basic" in lower:
        topics.append("basic_plan")
    if "pro" in lower:
        topics.append("pro_plan")
    if "price" in lower or "pricing" in lower or "cost" in lower or "plan" in lower:
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


def _is_yes(text: str) -> bool:
    return text.strip().lower() in {
        "yes", "y", "yeah", "yep", "sure", "ok", "okay", "sounds good", "let's do it", "lets do it"
    }


def _is_no(text: str) -> bool:
    return text.strip().lower() in {
        "no", "n", "nope", "not now", "later", "maybe later", "not yet"
    }


def _is_generic_plan_query(user_text: str) -> bool:
    """Detect broad plan questions (not plan-specific)."""
    lower = user_text.lower()
    asks_plan = "plan" in lower or "plans" in lower
    asks_pricing = "pricing" in lower or "price" in lower
    mentions_specific = "basic" in lower or "pro" in lower
    return (asks_plan or asks_pricing) and not mentions_specific


def _canonical_plans_response() -> str:
    """Deterministic plan response to prevent model hallucinations."""
    return (
        "AutoStream currently offers two plans:\n\n"
        "1) Basic Plan — $29/month\n"
        "- 10 videos per month\n"
        "- 720p exports\n"
        "- Basic auto-cut and trim\n"
        "- Standard email support\n\n"
        "2) Pro Plan — $79/month\n"
        "- Unlimited videos per month\n"
        "- 4K exports\n"
        "- AI-generated captions\n"
        "- Priority 24/7 live support\n"
        "- Advanced AI scene detection\n"
        "- Custom branding watermarks\n\n"
        "If you want, I can help you compare Basic vs Pro based on your workflow."
    )


def _strip_redundant_greeting(user_text: str, ai_text: str) -> str:
    """Remove leading greeting from AI output when user did not greet."""
    if classify_intent_heuristic(user_text) == Intent.GREETING:
        return ai_text
    return re.sub(r"^(hi|hello|hey there|hi there)[!.\s,]*", "", ai_text, flags=re.IGNORECASE).strip()


def _is_explicit_start_now(text: str) -> bool:
    """Detect explicit immediate start language to reduce friction."""
    lower = text.strip().lower()
    return lower in {
        "start", "start now", "yes start", "lets start", "let's start",
        "sign me up", "get started", "proceed"
    }


def _friendly_llm_fallback(err: Exception) -> str:
    """Return a graceful, user-facing fallback for transient LLM failures."""
    lower = str(err).lower()
    if "resource_exhausted" in lower or "quota exceeded" in lower or "429" in lower:
        return (
            "I hit a temporary capacity limit just now. I can still help with quick factual guidance: "
            "Basic is $29/month and Pro is $79/month. Ask me for pricing, features, policies, or a comparison."
        )
    return "I hit a temporary issue generating that response. Could you rephrase what you'd like to know?"


def _is_closure_ack(text: str) -> bool:
    """Detect short closure/acknowledgment replies after lead capture."""
    return text.strip().lower() in {
        "yes", "yea", "yeah", "yep", "ok", "okay", "cool", "great", "nice",
        "thanks", "thank you", "thankyou", "got it", "alright", "sure"
    }


def _plan_switch_response(target_plan: str) -> str:
    """Return a concise comparison-style response when user switches plans."""
    if target_plan == "basic_plan":
        return (
            "Sure. Basic is $29/month and includes 10 videos/month, 720p exports, "
            "basic auto-cut and trim, and standard email support. Compared to Pro, "
            "it is lower cost but has usage and feature limits. Want a quick side-by-side comparison?"
        )
    return (
        "Sure. Pro is $79/month and includes unlimited videos, 4K exports, AI captions, "
        "priority 24/7 support, advanced scene detection, and custom branding. Compared to Basic, "
        "it is better for higher-volume usage. Want a quick side-by-side comparison?"
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

llm = None  # initialized lazily

def _get_kb_context(query: str) -> str:
    try:
        return retrieve(query)
    except Exception:
        return get_full_kb_summary()


def node_respond(state: AgentState) -> AgentState:
    """Core LLM response node. Handles greetings, product questions, and intent detection."""
    global llm
    if llm is None:
        llm = _get_llm()

    messages = state["messages"]
    lead_info = state.get("lead_info", {})
    awaiting = state.get("awaiting_field")
    explained_topics = state.get("explained_topics", [])
    awaiting_signup_confirmation = state.get("awaiting_signup_confirmation", False)

    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    # --- Lightweight local path after successful lead capture ---
    if state.get("lead_captured") and _is_closure_ack(last_human):
        name = (lead_info.get("name") or "there").strip()
        return {
            **state,
            "messages": [AIMessage(content=f"You're welcome, {name}! If you need anything else, just ask.")],
            "intent": "other",
        }

    # --- Explicit confirmation gate before entering lead qualification ---
    if not awaiting and _is_explicit_start_now(last_human):
        return {
            **state,
            "messages": [AIMessage(content="__START_QUALIFICATION__")],
            "awaiting_signup_confirmation": False,
            "intent": "high_intent",
        }

    if awaiting_signup_confirmation:
        if _is_yes(last_human):
            return {
                **state,
                "messages": [AIMessage(content="__START_QUALIFICATION__")],
                "awaiting_signup_confirmation": False,
                "intent": "high_intent",
            }
        if _is_no(last_human):
            return {
                **state,
                "messages": [AIMessage(content="No problem. I can help you compare plans, pricing, and policies first. What would you like to explore?")],
                "awaiting_signup_confirmation": False,
                "intent": "product_inquiry",
            }
        return {
            **state,
            "messages": [AIMessage(content="Quick check: would you like to start signup now? Please reply yes or no.")],
            "awaiting_signup_confirmation": True,
            "intent": "other",
        }

    # --- If we're in field-collection mode, try to extract the value first ---
    if awaiting and messages:
        # If user asks a product question during lead collection, answer it first and resume.
        if classify_intent_heuristic(last_human) == Intent.PRODUCT_INQUIRY:
            kb_context = _get_kb_context(last_human)
            system_content = _build_system_prompt(kb_context, lead_info)
            lc_messages = [SystemMessage(content=system_content)] + [
                m for m in messages if isinstance(m, (HumanMessage, AIMessage))
            ]
            try:
                response = llm.invoke(lc_messages)
                raw_text = response.content if hasattr(response, "content") else str(response)
                clean_text, _ = _parse_intent_from_response(raw_text)
                clean_text = _strip_redundant_greeting(last_human, clean_text)
                if not clean_text.strip():
                    clean_text = "Could you clarify what you'd like to know more about: plans, pricing, features, or policies?"
            except Exception as err:
                clean_text = _friendly_llm_fallback(err)
            followup_map = {
                "name": "When you're ready, could you share your name?",
                "email": "When you're ready, could you share your email address?",
                "platform": "When you're ready, which platform do you mainly create on?"
            }
            reply = f"{clean_text}\n\n{followup_map.get(awaiting, 'When you are ready, we can continue signup details.')}"
            return {
                **state,
                "messages": [AIMessage(content=reply)],
                "intent": "product_inquiry",
                "awaiting_field": awaiting,
            }

        extracted = _extract_field(awaiting, last_human)
        if extracted:
            lead_info = {**lead_info, awaiting: extracted}
            # Determine next missing field
            next_field = _next_missing_field(lead_info)
            if next_field:
                prompt_map = {
                    "name": "Thanks! Could you share your name?",
                    "email": f"Great, {lead_info.get('name', '')}! What's your email address?",
                    "platform": f"Almost there! Which platform do you mainly create content on? (YouTube, Instagram, TikTok, etc.)"
                }
                reply = prompt_map[next_field]
                return {
                    **state,
                    "messages": [AIMessage(content=reply)],
                    "lead_info": lead_info,
                    "awaiting_field": next_field,
                    "intent": "high_intent",
                    "explained_topics": explained_topics,
                }
            else:
                # All fields collected — trigger tool
                return {
                    **state,
                    "messages": [AIMessage(content="__TRIGGER_LEAD_CAPTURE__")],
                    "lead_info": lead_info,
                    "awaiting_field": None,
                    "intent": "high_intent",
                    "explained_topics": explained_topics,
                }
        else:
            retry_map = {
                "email": "That does not look like a valid email yet. Please share it in a format like name@example.com.",
                "platform": "Got it. Which platform do you mainly create on? (YouTube, Instagram, TikTok, etc.)",
                "name": "Could you share your name as you'd like us to address you?"
            }
            return {
                **state,
                "messages": [AIMessage(content=retry_map.get(awaiting, "Could you share that detail again?"))],
                "intent": "high_intent",
                "awaiting_field": awaiting,
                "lead_info": lead_info,
                "explained_topics": explained_topics,
            }

    current_topics = _extract_topics(last_human)
    if _is_generic_plan_query(last_human):
        return {
            **state,
            "messages": [AIMessage(content=_canonical_plans_response())],
            "intent": "product_inquiry",
            "lead_info": lead_info,
            "awaiting_field": awaiting,
            "explained_topics": list(dict.fromkeys(explained_topics + ["basic_plan", "pro_plan", "pricing"])),
        }

    if "basic_plan" in current_topics and "pro_plan" in explained_topics and "basic_plan" not in explained_topics:
        return {
            **state,
            "messages": [AIMessage(content=_plan_switch_response("basic_plan"))],
            "intent": "product_inquiry",
            "lead_info": lead_info,
            "awaiting_field": awaiting,
            "explained_topics": list(dict.fromkeys(explained_topics + ["basic_plan"])),
        }
    if "pro_plan" in current_topics and "basic_plan" in explained_topics and "pro_plan" not in explained_topics:
        return {
            **state,
            "messages": [AIMessage(content=_plan_switch_response("pro_plan"))],
            "intent": "product_inquiry",
            "lead_info": lead_info,
            "awaiting_field": awaiting,
            "explained_topics": list(dict.fromkeys(explained_topics + ["pro_plan"])),
        }

    repeated_topics = [t for t in current_topics if t in explained_topics]
    if repeated_topics and len(current_topics) == 1:
        topic_name = repeated_topics[0].replace("_", " ")
        return {
            **state,
            "messages": [AIMessage(content=f"We've already covered the {topic_name}. I can give a quick recap or compare it with another plan if you'd like.")],
            "intent": "product_inquiry",
            "lead_info": lead_info,
            "awaiting_field": awaiting,
            "explained_topics": explained_topics,
        }

    # --- Normal LLM call ---
    kb_context = _get_kb_context(last_human)
    system_content = _build_system_prompt(kb_context, lead_info)

    # Build message list for LLM (system + history)
    lc_messages = [SystemMessage(content=system_content)] + [
        m for m in messages if isinstance(m, (HumanMessage, AIMessage))
    ]

    try:
        response = llm.invoke(lc_messages)
        raw_text = response.content if hasattr(response, "content") else str(response)
        clean_text, intent_str = _parse_intent_from_response(raw_text)
        clean_text = _strip_redundant_greeting(last_human, clean_text)
        if not clean_text.strip():
            clean_text = "Could you clarify what you'd like to know more about: plans, pricing, features, or policies?"
    except Exception as err:
        return {
            **state,
            "messages": [AIMessage(content=_friendly_llm_fallback(err))],
            "intent": "product_inquiry",
            "lead_info": lead_info,
            "awaiting_field": awaiting,
            "explained_topics": explained_topics,
            "awaiting_signup_confirmation": awaiting_signup_confirmation,
        }

    # Heuristic reconciliation: avoid premature lead qualification on exploratory language.
    heuristic_intent = classify_intent_heuristic(last_human)
    if heuristic_intent == Intent.HIGH_INTENT and intent_str in ("other", "greeting"):
        intent_str = "high_intent"
    elif heuristic_intent == Intent.PRODUCT_INQUIRY and intent_str == "high_intent":
        intent_str = "product_inquiry"

    new_topics = _extract_topics(last_human)
    merged_topics = list(dict.fromkeys(explained_topics + new_topics))

    new_state = {
        **state,
        "messages": [AIMessage(content=clean_text)],
        "intent": intent_str,
        "lead_info": lead_info,
        "awaiting_field": awaiting,
        "explained_topics": merged_topics,
        "awaiting_signup_confirmation": awaiting_signup_confirmation,
    }
    return new_state


def node_qualify_lead(state: AgentState) -> AgentState:
    """Starts the lead qualification flow by asking for the first missing field."""
    lead_info = state.get("lead_info", {})
    next_field = _next_missing_field(lead_info)

    if not next_field:
        # All collected, proceed to capture
        return {**state, "awaiting_field": None}

    prompt_map = {
        "name": (
            "That's great to hear! I'd love to get you set up. "
            "Could I start with your name?"
        ),
        "email": f"Thanks, {lead_info.get('name', '')}! What's your email address?",
        "platform": "And which platform do you primarily create content on? (YouTube, Instagram, TikTok, etc.)"
    }
    reply = prompt_map[next_field]
    return {
        **state,
        "messages": [AIMessage(content=reply)],
        "awaiting_field": next_field,
        "intent": "high_intent",
    }


def node_capture_lead(state: AgentState) -> AgentState:
    """Calls mock_lead_capture and sends a confirmation message."""
    if state.get("lead_captured"):
        return state  # idempotency guard

    lead = state.get("lead_info", {})
    result = mock_lead_capture(
        name=lead.get("name", "Unknown"),
        email=lead.get("email", "Unknown"),
        platform=lead.get("platform", "Unknown"),
    )

    confirmation = (
        f"🎉 You're all set, {lead.get('name', '')}! "
        f"We've captured your details and our team will reach out to your {lead.get('platform', '')} "
        f"creator account shortly. Welcome to AutoStream Pro! "
        f"Is there anything else I can help you with?"
    )
    return {
        **state,
        "messages": [AIMessage(content=confirmation)],
        "lead_captured": True,
        "awaiting_field": None,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _next_missing_field(lead_info: dict) -> Optional[str]:
    for field in ["name", "email", "platform"]:
        if not lead_info.get(field):
            return field
    return None


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_respond(state: AgentState) -> str:
    if state.get("lead_captured"):
        return END

    last_ai = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), ""
    )
    if last_ai == "__TRIGGER_LEAD_CAPTURE__":
        return "capture_lead"
    if last_ai == "__START_QUALIFICATION__":
        return "qualify_lead"

    intent = state.get("intent", "other")
    awaiting = state.get("awaiting_field")
    awaiting_signup_confirmation = state.get("awaiting_signup_confirmation", False)

    if intent == "high_intent" and not awaiting:
        lead_info = state.get("lead_info", {})
        if not awaiting_signup_confirmation and _next_missing_field(lead_info) is not None:
            return "confirm_signup"
        if _next_missing_field(lead_info) is None:
            return "capture_lead"
        return "qualify_lead"

    return END


def route_after_qualify(state: AgentState) -> str:
    if _next_missing_field(state.get("lead_info", {})) is None:
        return "capture_lead"
    return END


def node_confirm_signup(state: AgentState) -> AgentState:
    """Ask for explicit confirmation before starting lead collection."""
    return {
        **state,
        "messages": [AIMessage(content="Happy to help you sign up. Would you like to start signup now? (yes/no)")],
        "awaiting_signup_confirmation": True,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("respond",      node_respond)
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
        }
    )
    graph.add_edge("confirm_signup", END)
    graph.add_conditional_edges(
        "qualify_lead",
        route_after_qualify,
        {
            "capture_lead": "capture_lead",
            END: END,
        }
    )
    graph.add_edge("capture_lead", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public helper: run one turn
# ---------------------------------------------------------------------------

def run_turn(graph, state: AgentState, user_input: str) -> tuple[AgentState, str]:
    """
    Feed one user message into the graph and return (new_state, agent_reply).
    """
    new_state = graph.invoke({
        **state,
        "messages": state["messages"] + [HumanMessage(content=user_input)],
    })
    # Get last AI message
    reply = next(
        (m.content for m in reversed(new_state["messages"]) if isinstance(m, AIMessage)),
        ""
    )
    return new_state, reply


def initial_state() -> AgentState:
    return AgentState(
        messages=[],
        intent="other",
        lead_info={},
        lead_captured=False,
        awaiting_field=None,
        awaiting_signup_confirmation=False,
        explained_topics=[],
    )
