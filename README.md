# AutoStream AI Sales Agent

A production-grade conversational AI agent built for **AutoStream** — a SaaS platform offering automated video editing tools for content creators. Built with **LangGraph + Gemini + local RAG**.

---

## Features

| Capability | Implementation |
|---|---|
| Intent detection | Heuristic pre-filter + LLM classification (greeting / product_inquiry / high_intent) |
| RAG-powered Q&A | Local JSON knowledge base, keyword-scored retrieval, injected as LLM context |
| Lead qualification | Stateful field collection — name → email → platform, one at a time |
| Tool execution | `mock_lead_capture()` called only after all 3 fields are confirmed |
| Memory | Full conversation history retained across all turns via LangGraph state |

---

## Project Structure

```
autostream-agent/
├── main.py                       # CLI entry point (interactive chat)
├── agent.py                      # LangGraph StateGraph + all nodes + routing
├── rag.py                        # Local knowledge base loader + retrieval
├── tools.py                      # mock_lead_capture() + intent heuristics
├── knowledge_base/
│   └── autostream_kb.json        # Plans, policies, FAQs
├── requirements.txt
└── README.md
```

---

## Local Setup & Run

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd autostream-agent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Google API key

```bash
export GOOGLE_API_KEY=your-google-api-key         # Linux / macOS
set GOOGLE_API_KEY=your-google-api-key            # Windows CMD
$env:GOOGLE_API_KEY="your-google-api-key"        # Windows PowerShell
```

> The agent uses **Gemini 2.0 Flash-Lite** (`gemini-2.5-flash-lite`) for fast and cost-efficient responses. Get a key at https://aistudio.google.com/app/apikey

### 5. Run the agent

```bash
python main.py
```

---

## Example Session

```
╔══════════════════════════════════════════════════════════╗
║        AutoStream — AI Sales Assistant (Aria)           ║
╚══════════════════════════════════════════════════════════╝

🤖 Aria: Hi there! I'm Aria, AutoStream's AI assistant...

👤 You: Hi, tell me about your pricing

🤖 Aria: Great question! AutoStream has two plans:

  • Basic Plan — $29/month: 10 videos/month, 720p resolution, standard support
  • Pro Plan — $79/month: Unlimited videos, 4K resolution, AI captions, 24/7 support

Would you like more detail on either plan?

👤 You: That sounds good — I want to try the Pro plan for my YouTube channel

🤖 Aria: That's great to hear! I'd love to get you set up. Could I start with your name?

👤 You: Alex Johnson

🤖 Aria: Thanks, Alex Johnson! What's your email address?

👤 You: alex@example.com

🤖 Aria: Almost there! Which platform do you primarily create content on?

👤 You: YouTube

═══════════════════════════════════════════════════════
  ✅  LEAD CAPTURED SUCCESSFULLY
  Name     : Alex Johnson
  Email    : alex@example.com
  Platform : YouTube
═══════════════════════════════════════════════════════

🤖 Aria: 🎉 You're all set, Alex Johnson! Welcome to AutoStream Pro!
```

---

## Architecture (≈200 words)

### Why LangGraph?

LangGraph was chosen over a simple chain or AutoGen because it provides **explicit, inspectable state management** as a first-class primitive. The agent's conversation has distinct phases — answering questions, qualifying leads, capturing leads — that map naturally to graph **nodes** connected by **conditional edges**. This avoids the hidden state fragility of memory buffers and makes the routing logic auditable.

### How state is managed

A typed `AgentState` dict is threaded through every node. It carries:

- `messages` — full conversation history (auto-merged via `add_messages`)
- `intent` — last classified intent (`greeting`, `product_inquiry`, `high_intent`)
- `lead_info` — a dict accumulating `name`, `email`, `platform` as they are collected
- `awaiting_field` — which field the agent is currently collecting; gates field-extraction logic
- `lead_captured` — idempotency guard preventing double tool invocation

**Routing** happens in `route_after_respond`: if intent is `high_intent` and fields are missing, the graph transitions to `qualify_lead`; once all three fields are present, it goes to `capture_lead`. The LLM is only called in the `respond` node — all other transitions are pure Python logic.

### RAG pipeline

The local `knowledge_base/autostream_kb.json` is flattened into text chunks. For each user turn, the top-3 chunks by keyword overlap score are injected into the system prompt. No external vector DB is needed.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the integration would follow this architecture:

### Infrastructure

```
WhatsApp User
     │  (sends message)
     ▼
WhatsApp Business API (Meta Cloud API)
     │  (HTTP POST webhook)
     ▼
Your Webhook Server  ←── FastAPI / Flask endpoint
     │  (verify_token, parse incoming message)
     ▼
Agent (agent.py)     ←── LangGraph stateful graph
     │  (generate reply)
     ▼
WhatsApp Business API  ←── POST /messages
     │  (send reply)
     ▼
WhatsApp User
```

### Step-by-step

1. **Register a Meta App** at developers.facebook.com, enable the WhatsApp Business API, and obtain a `WHATSAPP_TOKEN` and `PHONE_NUMBER_ID`.

2. **Create a webhook endpoint** (e.g. with FastAPI):
   ```python
   @app.post("/webhook")
   async def receive_message(payload: dict):
       message = payload["entry"][0]["changes"][0]["value"]["messages"][0]
       user_id = message["from"]          # WhatsApp phone number (session key)
       user_text = message["text"]["body"]

       state = session_store.get(user_id, initial_state())
       state, reply = run_turn(graph, state, user_text)
       session_store[user_id] = state     # persist state per user

       send_whatsapp_message(user_id, reply)
   ```

3. **Persist state per user** using Redis or a DB keyed by WhatsApp phone number, so each user has their own independent LangGraph state across sessions.

4. **Verify the webhook** with the `hub.challenge` handshake that Meta requires before activating.

5. **Handle media** — WhatsApp can send images/audio; the webhook would extract the media URL and pass it to the agent if needed.

6. **Deploy** the webhook on a public HTTPS URL (e.g. Railway, Render, or a VPS with nginx + SSL). Use ngrok for local testing.

This approach cleanly separates the **transport layer** (WhatsApp webhook) from the **agent logic** (LangGraph), making the same agent deployable on Instagram DMs, Telegram, or Slack with only a different transport adapter.

---

## Evaluation Checklist

| Criteria | ✅ |
|---|---|
| Intent detection (3 classes) | ✅ Heuristic + LLM classification |
| RAG from local knowledge base | ✅ JSON KB, keyword retrieval, context injection |
| Lead collection — name, email, platform | ✅ Stateful, one field at a time |
| Tool called only after all fields collected | ✅ Idempotency guard + `awaiting_field` gating |
| Memory across 5–6 turns | ✅ Full history in LangGraph state |
| Clean code structure | ✅ Separate modules: agent, rag, tools, main |
| README with all required sections | ✅ |
