import os

from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        input="Hello",
    )

    print(response.output_text)
