---
title: Email Research Environment Server
emoji: 📬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Email Research Env

Email Research Env is a compact OpenEnv benchmark for multi-turn retrieval over a company inbox. The agent must search emails, open the right messages, and return a grounded answer with supporting email IDs.

This environment is designed for exactly the behavior we want to evaluate in agentic systems:

- iterative search refinement
- evidence-driven answers
- multi-step tool use over a local database
- reproducible grading with deterministic gold citations

## Why This Env

Searching enterprise email is a real workflow people already do manually. Questions like these come up all the time:

- Who approved this exception?
- When was the meeting moved to?
- Which person confirmed the final launch date?

This environment turns that workflow into a small, reproducible benchmark with easy, medium, and hard tasks.

## Action Space

The environment exposes 3 tools through a single typed action model:

### `search_emails`

Fields:

- `query`
- `top_k`
- `sent_after`
- `sent_before`
- `sender`

Behavior:

- runs SQLite FTS5 search over subject and body
- applies metadata filters for date and sender
- returns compact snippets only

### `read_email`

Fields:

- `email_id`

Behavior:

- returns the full email body and metadata for one message

### `return_final_answer`

Fields:

- `answer`
- `cited_email_ids`

Behavior:

- submits the final answer
- grades answer correctness, evidence quality, and efficiency
- ends the episode

## Observation Space

Each observation can include:

- `question`
- `last_action`
- `search_results`
- `email`
- `tool_message`
- `reward_breakdown`
- `steps_used`
- `steps_remaining`
- inherited `reward`, `done`, and `metadata`

## Tasks

The benchmark includes 3 official tasks:

1. `task_easy_atlas_owner`
   Ask who confirmed the final Atlas launch date. One key email contains the answer.

2. `task_medium_redwood_reschedule`
   Ask for the new Redwood vendor demo date and who requested the change. The model must combine two emails.

3. `task_hard_northwind_approval`
   Ask who gave final approval for the Northwind pricing exception and on what date. The model must find a clue in one email, then refine search to find the approval email.

The medium and hard tasks are intentionally multi-email. The hard task is intentionally multi-hop.

## Reward Model

Rewards stay in the `0.0-1.0` range and are mostly terminal:

- `0.65` answer correctness
- `0.2` citation correctness
- `0.15` efficiency

There are also small shaping rewards for useful reads:

- first read of a gold email gives partial progress
- repeated identical searches and repeated email reads reduce the efficiency component
- using more of the available step budget reduces the final efficiency score

## Data and Search Design

The environment uses:

- `data/email_blueprints.json` for the synthetic email corpus
- `data/tasks.json` for benchmark definitions and gold answers
- `data/inbox.db` for the searchable SQLite inbox

SQLite is only used for the email corpus. Task definitions stay in JSON for clarity and easy editing.

Search is backed by:

- SQLite FTS5 on `subject` and `body`
- regular indexes on `sent_at` and `sender`

## Generating the Inbox

The inbox is synthetic and task-first.

We do not ship a giant raw mailbox. Instead, we keep only:

- emails needed for the 3 benchmark tasks
- distractor emails that create realistic ambiguity

Rebuild the DB locally with:

```bash
python scripts/build_inbox_db.py
```

## Quick Start

```python
from my_env import EmailAction, MyEnv

with MyEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="task_easy_atlas_owner")
    print(result.observation.question)

    result = env.step(
        EmailAction(
            action_type="search_emails",
            query="Atlas launch date",
            top_k=3,
        )
    )
    print(result.observation.search_results)
```

## Running Locally

Install dependencies:

```bash
uv sync
```

Build the inbox DB:

```bash
python scripts/build_inbox_db.py
```

Run the server:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Docker

Build:

```bash
docker build -t email-research-env:latest -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 8000:8000 email-research-env:latest
```

## Baseline Inference

The baseline script is `inference.py` at the repo root.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional runtime variables:

- `LOCAL_IMAGE_NAME` if you want `inference.py` to launch a local Docker image explicitly
- `ENV_URL` if you want to point the baseline at an already-running local server instead of Docker

For submission, keep the variable names exactly as above even if you use another OpenAI-compatible provider. For example, if you use OpenRouter, set:

- `API_BASE_URL=https://openrouter.ai/api/v1`
- `MODEL_NAME=<your-openrouter-model>`
- `HF_TOKEN=<your-openrouter-api-key>`

Run:

```bash
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4.1-mini \
HF_TOKEN=... \
LOCAL_IMAGE_NAME=email-research-env:latest \
python inference.py
```

The script emits the required structured logs:

- `[START]`
- `[STEP]`
- `[END]`

## Baseline Scores

Example reproducible baseline from a real local run on 2026-04-08:

- provider: OpenRouter
- model: `z-ai/glm-5`
- easy: `0.97`
- medium: `0.12`
- hard: `0.89`

Recompute these if you change the submission model.

## Project Structure

```text
my_env/
├── README.md
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── client.py
├── models.py
├── data/
│   ├── email_blueprints.json
│   ├── tasks.json
│   └── inbox.db
├── scripts/
│   └── build_inbox_db.py
└── server/
    ├── app.py
    ├── inbox_repository.py
    └── my_env_environment.py
```
