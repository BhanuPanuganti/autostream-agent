# AutoStream AI Sales Agent

AutoStream AI Sales Agent is a conversational assistant for product Q&A, plan guidance, and lead qualification. It is built with LangGraph, Gemini, and a local RAG layer over a JSON knowledge base.

## Features

| Capability | Implementation |
|---|---|
| Intent detection | Heuristic pre-filter + LLM classification (`greeting`, `product_inquiry`, `high_intent`) |
| RAG-powered answers | Local JSON knowledge base, keyword-scored retrieval, prompt-grounded responses |
| Lead qualification | Stateful field collection (`name -> email -> platform`) with validation |
| Tool execution | `mock_lead_capture()` is called only after all required fields are collected |
| Conversation memory | Full turn history and control flags retained in LangGraph state |
| Support recovery flow | Deterministic handling for dissatisfaction, plan switch questions, and refund-policy guidance |

## Project Structure

```text
autostream-agent/
|- main.py
|- agent.py
|- rag.py
|- tools.py
|- knowledge_base/
|  |- autostream_kb.json
|- requirements.txt
|- README.md
```

## How To Run Locally

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd autostream-agent
```

If you received this project as a zip file, extract it and run commands from the extracted `autostream-agent` directory.

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

Linux/macOS:

```bash
export GOOGLE_API_KEY=your-google-api-key
```

Windows CMD:

```cmd
set GOOGLE_API_KEY=your-google-api-key
```

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY="your-google-api-key"
```

Optional model override:

```bash
export GOOGLE_MODEL=gemini-2.5-flash-lite
```

### 5. Run the assistant

```bash
python main.py
```

## Example Session

```text
Aria: Hi there. I can help with pricing, features, policies, and plan guidance.

You: I publish daily on YouTube and need captions. Which plan fits me?

Aria: Based on that workflow, Pro is the better fit. It includes unlimited videos,
4K exports, AI-generated captions, advanced scene detection, custom branding,
and priority 24/7 support. You can start with the 7-day Pro trial with no credit card required.
Would you like to start with Pro now?

You: Yes, start

Aria: Great. Could I start with your name?

You: Alex Johnson
You: alex@example.com
You: YouTube

System: LEAD CAPTURED SUCCESSFULLY
```

## Architecture Explanation (Approx. 200 Words)

LangGraph was selected because this agent is a controlled conversation, not a single prompt-response call. The flow has clear stages: answer product questions, detect buying intent, collect lead details, and trigger lead capture only after required fields are complete. LangGraph maps this cleanly using nodes and conditional edges, which makes behavior explicit and easier to debug.

AutoGen was considered, but LangGraph gives tighter deterministic control for this assignment. The rubric emphasizes correctness and predictable tool usage. With graph-based routing, those constraints are implemented directly in code rather than left to prompt behavior alone.

State is managed through a typed `AgentState` object passed across nodes. It includes message history, current intent, lead fields, collection progress (`awaiting_field`), and idempotency flags (`lead_captured`). Routing functions inspect state and decide transitions such as `respond -> confirm_signup -> qualify_lead -> capture_lead`. The LLM is used for natural language generation and intent support, while business rules remain in Python.

RAG is local and lightweight: the JSON knowledge base is flattened into chunks, scored against the query, and relevant excerpts are injected into the system prompt. This keeps responses grounded without requiring external vector infrastructure.

## WhatsApp Deployment Using Webhooks

To integrate this agent with WhatsApp, keep channel transport separate from agent logic.

1. Create a Meta app and enable WhatsApp Business Cloud API.
2. Configure webhook URL, verification token, and app credentials.
3. Build a webhook endpoint (FastAPI or Flask) to receive message events.
4. Parse sender phone number and text from inbound payloads.
5. Use sender number as session key, load state, run one agent turn, and persist updated state.
6. Send the generated reply through the WhatsApp messages endpoint.
7. Verify webhook signatures and implement retry-safe processing.

Suggested production controls:

- Persist session state in Redis or a database keyed by phone number
- Add idempotency keys for redelivered webhook events
- Add structured logging, monitoring, and alerting
- Rate-limit inbound requests and secure secrets through environment configuration
- Run behind HTTPS with proper access controls

This adapter pattern allows the same core agent to be reused across other messaging channels with minimal changes.

## Assumptions and Limitations

### Assumptions

- The runtime has valid Gemini API access through `GOOGLE_API_KEY`.
- The knowledge source is the local file `knowledge_base/autostream_kb.json`.
- Lead capture is represented by `mock_lead_capture()` and can be replaced with a real CRM API integration.

### Current Limitations

- Retrieval uses lightweight keyword overlap scoring, not semantic embeddings.
- Lead capture is a mock implementation and does not persist to an external system.
- Webhook deployment is described architecturally; production deployment requires infrastructure provisioning and Meta app configuration.
- The agent is intentionally constrained to known plan and policy data and will not infer unsupported commercial claims.

