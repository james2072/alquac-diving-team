"""
test_connection.py – Verify connection to Google AI Studio or configured LLM provider.
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from configs.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from rag_retrieval.llm_client import chat


def main() -> None:
    print("=" * 50)
    print("        LLM Connection Playground")
    print("=" * 50)
    print(f"Target URL : {LLM_BASE_URL}")
    print(f"Model      : {LLM_MODEL}")

    masked_key = (
        f"{LLM_API_KEY[:6]}...{LLM_API_KEY[-4:]}"
        if len(LLM_API_KEY) > 10
        else "Not Set or Too Short"
    )
    print(f"API Key    : {masked_key}")
    print("=" * 50)

    if not LLM_API_KEY or LLM_API_KEY == "AIza-your-google-ai-studio-key":
        print("\n❌ ERROR: Please define a valid API key in your .env file at the root folder.\n")
        sys.exit(1)

    print("Sending test prompt: 'Hello, briefly introduce yourself.'...")

    try:
        response = chat("Hello, briefly introduce yourself.", max_tokens=100)
        print("\n✅ SUCCESS! Connection verified successfully.\n")
        print("Response from LLM:")
        print("-" * 50)
        print(response.strip())
        print("-" * 50)
    except Exception as e:
        print("\n❌ ERROR: Connection failed!")
        print(f"Details: {e}\n")
        print("Please check your .env configuration (API key and Base URL validity).")
        sys.exit(1)


if __name__ == "__main__":
    main()
