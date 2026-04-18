# LiteLLM Proxy — Course Setup

Proxies OpenAI with per-student virtual keys, spend budgets, and rate limits.

## Quick start

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, LITELLM_MASTER_KEY (invent a secret), POSTGRES_PASSWORD
docker compose up -d
```

Proxy is now available at `http://localhost:4000`.  
Admin UI: `http://localhost:4000/ui` (login with master key).

## Generate student keys

```bash
# Option A: named list
echo -e "alice@example.com\nbob@example.com" > students.txt
LITELLM_MASTER_KEY=your-master-key python generate_keys.py --students students.txt

# Option B: 120 anonymous keys
LITELLM_MASTER_KEY=your-master-key python generate_keys.py --count 120
```

Keys are written to `keys_output.csv`. Distribute the `key` column to students.

## Student usage

Students point any OpenAI-compatible client at your proxy:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-their-personal-key",
    base_url="http://your-server:4000",
)
```

## Per-key limits (edit generate_keys.py to change)

| Setting | Value |
|---|---|
| Budget | $5 / 30 days |
| RPM | 20 req/min |
| Models | gpt-4o-mini, gpt-4o |

## Monitor usage

```bash
# All keys + spend
curl -H "Authorization: Bearer $LITELLM_MASTER_KEY" http://localhost:4000/key/list | jq .

# Single key info
curl -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  "http://localhost:4000/key/info?key=sk-student-key" | jq .
```
