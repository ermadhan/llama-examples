import argparse
import os
import sys

import requests

URL = "https://api.meta.ai/v1/responses"
DEFAULT_MODEL = "muse-wonder-1.0"
DEFAULT_PROMPT = (
    "Write a script that randomly generates a haiku about Meta, and explain "
    "your reasoning behind what script language you would choose."
)
KEY_VARS = ("MODEL_API_KEY", "LLAMA_API_KEY")


def get_api_key() -> str | None:
    for name in KEY_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def extract_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts = []
    for item in data.get("output", []):
        for block in item.get("content") or []:
            if block.get("type") in ("output_text", "text") and block.get("text"):
                parts.append(block["text"])
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the Meta Model API Responses endpoint.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="The prompt to send.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID (default: %(default)s).")
    parser.add_argument("--raw", action="store_true", help="Print the full JSON response.")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        parser.error("No API key found. Set MODEL_API_KEY (or LLAMA_API_KEY) in your environment.")

    payload = {
        "model": args.model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": args.prompt}],
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
    except requests.RequestException as exc:
        sys.exit(f"Request failed: {exc}")

    if not response.ok:
        sys.exit(f"API error {response.status_code}: {response.text}")

    data = response.json()
    print(data if args.raw else extract_text(data) or data)


if __name__ == "__main__":
    main()
