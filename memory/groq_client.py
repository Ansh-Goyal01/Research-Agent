import time
import re
from groq import Groq, RateLimitError
import config


def create_client():
    return Groq(api_key=config.GROQ_API_KEY)


def call_with_retry(client, messages, max_tokens=None, temperature=0.3, agent_name="agent"):
    max_tokens = max_tokens or config.MAX_TOKENS
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            error_msg = str(e)
            wait_seconds = _extract_wait_time(error_msg)

            if "tokens per day" in error_msg:
                print(f"\n[{agent_name}] Daily token limit reached.")
                print(f"[{agent_name}] Resets at 5:30 AM IST (midnight UTC)")
                print(f"[{agent_name}] Wait time: {wait_seconds} seconds ({round(wait_seconds/60, 1)} minutes)")
                _countdown(wait_seconds, agent_name)

            elif "tokens per minute" in error_msg:
                print(f"\n[{agent_name}] Per-minute limit hit. Waiting {wait_seconds}s...")
                _countdown(wait_seconds, agent_name)

            else:
                print(f"\n[{agent_name}] Rate limit hit. Waiting {wait_seconds}s...")
                _countdown(wait_seconds, agent_name)

            if attempt == max_attempts - 1:
                raise

        except Exception as e:
            if attempt < max_attempts - 1:
                wait = (attempt + 1) * 10
                print(f"[{agent_name}] Error: {str(e)[:100]}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"[{agent_name}] Failed after {max_attempts} attempts")


def _extract_wait_time(error_msg):
    patterns = [
        r"try again in (\d+(?:\.\d+)?)s",
        r"try again in (\d+(?:\.\d+)?) seconds",
        r"try again in (\d+)m(\d+(?:\.\d+)?)s",
        r"retry after (\d+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, error_msg, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                minutes = float(match.group(1))
                seconds = float(match.group(2))
                return int(minutes * 60 + seconds) + 5
            return int(float(match.group(1))) + 5
    return 65


def _countdown(seconds, agent_name):
    seconds = int(seconds)
    print(f"[{agent_name}] Resuming in: ", end="", flush=True)
    for remaining in range(seconds, 0, -10):
        print(f"{remaining}s... ", end="", flush=True)
        time.sleep(min(10, remaining))
    print("Retrying now.")