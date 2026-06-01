"""
Azure OpenAI sanity check.
Run: python test_azure.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Load config ──────────────────────────────────────────────────────────────
ENDPOINT = os.getenv("AZURE_EASTUS2_ENDPOINT", "").rstrip("/")
API_KEY = os.getenv("AZURE_EASTUS2_API_KEY", "")
API_VERSION = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")


def mask(s):
    if not s:
        return "<empty>"
    return s[:6] + "…" + s[-4:] if len(s) > 12 else "<short>"


print("=" * 60)
print("Azure OpenAI configuration check")
print("=" * 60)
print(f"  Endpoint    : {ENDPOINT or '<empty>'}")
print(f"  API Version : {API_VERSION}")
print(f"  Deployment  : {DEPLOYMENT}")
print(f"  API Key     : {mask(API_KEY)}  (length={len(API_KEY)})")
print()

if not all([ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT]):
    print("✗ One or more env vars are missing. Check your .env file.")
    sys.exit(1)


PAYLOAD = {
    "model": DEPLOYMENT,
    "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
    "max_completion_tokens": 50,
}


def run(name, headers):
    print(f"── Test: {name} " + "─" * (50 - len(name)))
    try:
        r = requests.post(
            f"{ENDPOINT}/chat/completions",
            params={"api-version": API_VERSION},
            headers={**headers, "Content-Type": "application/json"},
            json=PAYLOAD,
            timeout=30,
        )
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            print(f"  Reply : {content!r}")
            print(f"  ✓ PASS")
            return True
        else:
            try:
                err = r.json().get("error", {})
                print(f"  Error : {err.get('code', '?')} — {err.get('message', r.text[:200])}")
            except Exception:
                print(f"  Body  : {r.text[:200]}")
            print(f"  ✗ FAIL")
            return False
    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        return False
    finally:
        print()


# ─── Test 1: raw HTTP with api-key header (Azure's standard) ─────────────────
t1 = run("Raw HTTP — api-key header", {"api-key": API_KEY})

# ─── Test 2: raw HTTP with Bearer auth (what the OpenAI SDK sends) ───────────
t2 = run("Raw HTTP — Bearer auth", {"Authorization": f"Bearer {API_KEY}"})

# ─── Test 3: OpenAI SDK with our patched default_headers ─────────────────────
print("── Test: OpenAI SDK + default_headers patch " + "─" * 14)
try:
    from openai import OpenAI
    client = OpenAI(
        base_url=ENDPOINT,
        api_key=API_KEY,
        default_headers={"api-key": API_KEY},
        default_query={"api-version": API_VERSION},
    )
    resp = client.chat.completions.create(**PAYLOAD)
    print(f"  Reply : {resp.choices[0].message.content!r}")
    print(f"  ✓ PASS\n")
    t3 = True
except Exception as e:
    print(f"  ✗ FAIL: {e}\n")
    t3 = False

# ─── Test 4: OpenAI SDK without the patch (current app.py behavior) ──────────
print("── Test: OpenAI SDK — no patch (current code) " + "─" * 12)
try:
    from openai import OpenAI
    client = OpenAI(
        base_url=ENDPOINT,
        api_key=API_KEY,
        default_query={"api-version": API_VERSION},
    )
    resp = client.chat.completions.create(**PAYLOAD)
    print(f"  Reply : {resp.choices[0].message.content!r}")
    print(f"  ✓ PASS\n")
    t4 = True
except Exception as e:
    print(f"  ✗ FAIL: {e}\n")
    t4 = False


# ─── Verdict ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  Raw HTTP (api-key)     : {'✓' if t1 else '✗'}")
print(f"  Raw HTTP (Bearer)      : {'✓' if t2 else '✗'}")
print(f"  SDK with patch         : {'✓' if t3 else '✗'}")
print(f"  SDK no patch (current) : {'✓' if t4 else '✗'}")
print()

if not t1 and not t2:
    print("→ The key/endpoint combo is invalid.")
    print("  Check: key matches Azure portal, deployment 'gpt-5-mini' exists,")
    print("        endpoint matches resource (region + name spelling).")
elif t1 and not t4:
    print("→ The api-key header works but Bearer doesn't.")
    print("  Apply the `default_headers={'api-key': API_KEY}` patch to get_azure_client()")
    print("  and redeploy. Also paste the same fix to Streamlit Cloud (just push to GitHub).")
elif t4:
    print("→ Everything works locally. If the deployed app still 401s,")
    print("  the Streamlit Cloud secret doesn't match this local .env value.")
    print("  Re-paste the AZURE_EASTUS2_API_KEY into Streamlit Cloud Secrets.")