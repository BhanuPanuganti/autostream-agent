"""
main.py — CLI entry point for the AutoStream AI Agent.

Run:
    python main.py

Set your API key first:
    export GOOGLE_API_KEY=your-google-api-key
"""

import sys
import os
import re
import shutil
import textwrap

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from agent import build_graph, run_turn, initial_state


BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AutoStream — AI Sales Assistant (Aria)            ║
║       Powered by Gemini + LangGraph + Local RAG          ║
║       Type  'quit' or 'exit' to end the session.         ║
╚══════════════════════════════════════════════════════════╝
"""

ARIA_WELCOME = (
    "👋 Hi there! I'm Aria, AutoStream's AI assistant. "
    "I can help with plans, features, pricing, FAQs, and policies. "
    "You can ask things like: refund policy, free trial, billing cycle, cancellation, "
    "or supported publishing platforms. What can I help you with?"
)


COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_DIM = "\033[90m"


def _term_width() -> int:
    return max(72, shutil.get_terminal_size((100, 24)).columns)


def _wrap_lines(text: str, indent: str = "  ") -> str:
    width = max(60, _term_width() - len(indent) - 2)
    lines = []
    for para in text.splitlines() or [text]:
        if not para.strip():
            lines.append("")
            continue
        lines.append(
            textwrap.fill(
                para,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
                replace_whitespace=False,
            )
        )
    return "\n".join(lines)


def _hr() -> str:
    return "─" * min(96, _term_width())


def _print_assistant(text: str) -> None:
    print(f"\n{COLOR_CYAN}Aria{COLOR_RESET}")
    print(_wrap_lines(text))
    print(f"{COLOR_DIM}{_hr()}{COLOR_RESET}")


def _print_system(text: str) -> None:
    print(f"\n{COLOR_YELLOW}System{COLOR_RESET}")
    print(_wrap_lines(text))
    print(f"{COLOR_DIM}{_hr()}{COLOR_RESET}")


def _friendly_runtime_error(err: Exception) -> str:
    """Convert provider exceptions into actionable user-facing guidance."""
    msg = str(err)
    lower = msg.lower()

    if "resource_exhausted" in lower or "quota exceeded" in lower or "429" in lower:
        retry_match = re.search(r"retry in\s+([0-9.]+)s", msg, re.IGNORECASE)
        retry_hint = f" Retry after about {retry_match.group(1)}s." if retry_match else ""
        return (
            "Gemini API quota is exhausted for this project/key." + retry_hint + "\n"
            "What to do:\n"
            "1) Check usage and limits: https://ai.dev/rate-limit\n"
            "2) Check quota docs: https://ai.google.dev/gemini-api/docs/rate-limits\n"
            "3) Enable billing or use a project/key with available quota."
        )

    if "not_found" in lower or "404" in lower:
        return (
            "Configured Gemini model is not available for your account/API version.\n"
            "Set a supported model with GOOGLE_MODEL and retry."
        )

    if "api key" in lower or "permission_denied" in lower or "unauthorized" in lower:
        return (
            "Authentication/authorization issue with GOOGLE_API_KEY.\n"
            "Verify the key and ensure Gemini API access is enabled for this project."
        )

    return f"Unexpected runtime error: {msg}"


def main():
    print(BANNER)
    print(f"{COLOR_DIM}{_hr()}{COLOR_RESET}")

    try:
        graph = build_graph()
    except EnvironmentError as e:
        _print_system(f"Configuration error: {e}")
        sys.exit(1)

    state = initial_state()
    _print_assistant(ARIA_WELCOME)

    while True:
        try:
            user_input = input(f"{COLOR_GREEN}You{COLOR_RESET} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Thanks for chatting with AutoStream.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/clear", "clear"):
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            print(f"{COLOR_DIM}{_hr()}{COLOR_RESET}")
            _print_assistant("Screen cleared. What would you like to continue with?")
            continue
        if cmd in ("/help", "help"):
            _print_system("Commands: /clear to clear the screen, /help for this message, quit/exit to end chat.")
            continue

        if cmd in ("quit", "exit", "bye", "goodbye"):
            _print_assistant("Thanks for chatting! Feel free to reach out anytime. Goodbye! 👋")
            break

        try:
            state, reply = run_turn(graph, state, user_input)
        except Exception as e:
            _print_system(f"Error during agent turn: {_friendly_runtime_error(e)}")
            break

        _print_assistant(reply)

        # Session ends gracefully after lead is captured if user says bye
        if state.get("lead_captured"):
            try:
                followup = input(f"{COLOR_GREEN}You{COLOR_RESET} > ").strip()
                if not followup or followup.lower() in ("quit", "exit", "bye", "no", "nope", "thanks"):
                    _print_assistant("Great! Talk soon. Welcome aboard! 🚀")
                    break
                state, reply = run_turn(graph, state, followup)
                _print_assistant(reply)
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
