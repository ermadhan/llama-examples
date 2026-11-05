import argparse
import os
import sys

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
KEY_VARS = ("MODEL_API_KEY", "LLAMA_API_KEY")


def get_api_key() -> str | None:
    for name in KEY_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a prompt to the Meta Model API.")
    parser.add_argument(
        "--prompt",
        default="Hello, how are you?",
        help="The prompt to send (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="The model ID to use (default: %(default)s).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the available model IDs before sending the prompt.",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        parser.error(
            "No API key found. Set MODEL_API_KEY (or LLAMA_API_KEY) in your environment."
        )

    from openai import APIConnectionError, APIStatusError, OpenAI

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    try:
        if args.list_models:
            for model in client.models.list():
                print(model.id)

        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_completion_tokens=1024,
            temperature=0.7,
        )
    except APIStatusError as exc:
        hint = (
            " Check that the key is valid and was created at https://dev.meta.ai/api-keys."
            if exc.status_code == 401
            else ""
        )
        sys.exit(f"API error {exc.status_code}: {exc.message}.{hint}")
    except APIConnectionError as exc:
        sys.exit(f"Could not connect to {BASE_URL}: {exc}")

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
