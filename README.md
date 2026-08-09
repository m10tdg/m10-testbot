# TESTBOT — Local Development, Step by Step

Follow this top to bottom, in order. **Do not skip ahead** — each step proves the
previous one worked before you build on top of it. Every step tells you exactly
what command to run and exactly what you should see.

You will end up with **4 terminal windows open** (one per running service) plus
the one you're typing setup commands into. That's normal — this is what a
microservices system looks like when you run it locally.

---

## Prerequisites (check these first)

Open a terminal and run each of these. If any fail, install that tool before continuing.

```bash
docker --version        # need Docker installed and running
node --version           # need v20 or newer
python3 --version        # need 3.12 or newer (3.10+ probably works too)
```

You also need an **OpenAI API key** (the agents call `openai.com`'s API to generate
test scripts and analyze results). Get one at platform.openai.com if you don't have
one — you'll paste it into a `.env` file in Step 6.

---

## Step 1 — Start the infrastructure (Kafka, Postgres, Qdrant, fake S3)

```bash
cd testbot/infra
docker compose up -d
```

Wait about 20 seconds, then check everything is actually running:

```bash
docker compose ps
```

**You should see:** 6 containers, all with status `Up` — `postgres`, `zookeeper`,
`kafka`, `kafka-ui`, `qdrant`, `localstack`.

If any container is missing or restarting, run `docker compose logs <name>` to see why
before continuing.

---

## Step 2 — Verify the database schema loaded

```bash
docker exec -it infra-postgres-1 psql -U testbot -d testbot -c "\dt"
```

**You should see:** a list of tables — `tenants`, `projects`, `documents`, `test_runs`, `reports`.

Now check the seeded demo tenant exists:

```bash
docker exec -it infra-postgres-1 psql -U testbot -d testbot -c "SELECT id, name FROM tenants;"
```

**You should see:** one row — `Demo Company`, with id `11111111-1111-1111-1111-111111111111`.

Write that ID down (or just remember it's `1111...1111`) — you'll use it as `tenantId`
in every request from here on, since we haven't built login/auth yet.

**If step 2 doesn't show these tables:** the schema didn't auto-load. Tear down and
recreate the Postgres volume: `docker compose down -v && docker compose up -d`, then
repeat Step 2. (The `-v` wipes the database — that's fine, nothing important is in it yet.)

---

## Step 3 — Create the S3 buckets (fake S3, via Localstack)

```bash
cd testbot/infra
bash create-buckets.sh
```

**You should see:** two lines of (mostly empty) output with no errors.

Verify:

```bash
curl http://localhost:4566/m10-documents
curl http://localhost:4566/m10-artifacts
```

**You should see:** XML output starting with `<ListBucketResult>` for both — meaning the
bucket exists and is empty. If you see an error instead, re-run Step 3.

---

## Step 4 — Run Project Service, create your first real project

Open a **new terminal window** (keep it open — this service needs to keep running).

```bash
cd testbot/services/project-service
npm install
cp .env.example .env
npm start
```

**You should see:** `[project-service] listening on http://localhost:4001`

Leave that terminal running. In your **original terminal**, create a project:

```bash
curl -X POST http://localhost:4001/projects \
  -H "Content-Type: application/json" \
  -d '{"tenantId":"11111111-1111-1111-1111-111111111111","name":"Demo Project","slug":"demo-project"}'
```

**You should see:** a JSON response like:
```json
{"projectId":"a1b2c3d4-...","qdrantCollection":"11111111-1111-1111-1111-111111111111_a1b2c3d4-..."}
```

**Copy that `projectId` somewhere** — you'll need it in every step below. From here on,
this guide calls it `<PROJECT_ID>`.

Verify the Qdrant collection was really created:

```bash
curl http://localhost:6333/collections
```

**You should see:** your new collection name listed under `"collections"`.

This proves the full chain worked: HTTP request → Express route → Postgres insert →
Qdrant collection creation.

---

## Step 5 — Run Document Service, upload a document, watch it queue for embedding

New terminal window:

```bash
cd testbot/services/document-service
npm install
cp .env.example .env
npm start
```

**You should see:** `[document-service] listening on http://localhost:4002`

Create a tiny test document on your machine:

```bash
cd testbot
echo "Users can log in with their email and password.

Users can reset their password via a 'Forgot password' link on the login page.

After 3 failed login attempts, the account is locked for 15 minutes." > requirements.txt
```

Upload it (replace `<PROJECT_ID>` with the value from Step 4):

```bash
curl -X POST http://localhost:4002/documents/upload \
  -F "tenantId=11111111-1111-1111-1111-111111111111" \
  -F "projectId=<PROJECT_ID>" \
  -F "docType=requirement" \
  -F "file=@requirements.txt"
```

**You should see:** `{"documentId":"...","s3Path":"...","status":"uploaded"}`

**Check the document-service terminal** — you should NOT see any errors there. The
upload succeeded, the file is in fake S3, and a `document.uploaded` Kafka event was
published. Nothing has consumed it yet — that's the next step.

You can also watch the event land in Kafka visually: open **http://localhost:8085**
(Kafka UI) in your browser → Topics → `document.uploaded` → you should see 1 message.

---

## Step 6 — Run Embedding Service, watch it turn the document into vectors

```bash
cd testbot/services/embedding-service
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Now **edit `.env`** and paste your real OpenAI key into `OPENAI_API_KEY=`.

```bash
python main.py
```

**You should see immediately** (it processes the message that's been waiting since Step 5):
```
[embedding-service] waiting for document.uploaded events...
[embedding-service] processing document ... -> collection ...
[embedding-service] split into 1 chunk(s)
[embedding-service] wrote 1 vector(s) into '...'
```

Leave this terminal running — it keeps listening for more uploads.

Verify the vector really landed in Qdrant:

```bash
curl http://localhost:6333/collections/<QDRANT_COLLECTION>/points/count \
  -H "Content-Type: application/json" -d '{}'
```

(use the `qdrantCollection` value from Step 4's response)

**You should see:** `{"result":{"count":1},"status":"ok","time":...}`

This proves the full RAG ingestion chain works: upload → S3 → Kafka → chunk → embed →
Qdrant. This is what your AI agents will search when generating tests.

---

## Step 7 — Run the AI Agent Orchestrator (this is the part you most wanted to build)

```bash
cd testbot/agents/orchestrator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

Edit `.env` — paste your OpenAI key again, and make sure `DATABASE_URL` matches
Step 2's Postgres (`postgres://testbot:testbot@localhost:5432/testbot` — already
correct by default).

```bash
python main.py
```

**You should see:** `[orchestrator] waiting for test.requested events...`

Leave this running. This one process contains all five agents (scenario, execution,
visual analysis, analysis, reporting) as nodes in one LangGraph graph — nothing
happens until a `test.requested` event arrives, which is what the next step creates.

---

## Step 8 — Run CI Integration Service, trigger a real test run

```bash
cd testbot/services/ci-integration-service
npm install
cp .env.example .env
npm start
```

**You should see:** `[ci-integration-service] listening on http://localhost:4003`

Now, in your original terminal, trigger a run (replace `<PROJECT_ID>`):

```bash
curl -X POST http://localhost:4003/runs \
  -H "Content-Type: application/json" \
  -d '{
    "tenantId":"11111111-1111-1111-1111-111111111111",
    "projectId":"<PROJECT_ID>",
    "url":"https://example.com",
    "prompt":"Validate the homepage loads and shows a heading",
    "runSource":"ui"
  }'
```

**You should see:** `{"runId":"...","status":"queued"}`

**Copy that `runId`.**

---

## Step 9 — Watch the whole AI pipeline run, end to end

Switch to your **orchestrator terminal** (Step 7). Within a few seconds you should see:

```
=== starting run <runId> ===
[scenario-agent] generating script for run ...
[scenario-agent] generated N line(s) of test code
[execution-agent] running generated script for run ...
[execution-agent] done. passed=True, consoleErrors=0
[visual-analysis-agent] comparing screenshots for run ...
[visual-analysis-agent] no baseline found, saving this screenshot as the new baseline
[analysis-agent] analyzing results for run ...
[analysis-agent] severity=info
[reporting-agent] building report for run ...
[reporting-agent] report written to ...
=== run <runId> finished: completed ===
```

**This is the moment you're building toward** — an AI agent read your prompt, wrote
real Playwright test code, ran it against a real website in a real browser, took a
screenshot, and wrote a report, entirely on its own.

Now check the status via the API (replace `<runId>`):

```bash
curl http://localhost:4003/runs/<runId>
```

**You should see:** `"status":"completed"` and a `report` object with an `s3_path`.

To view the actual report HTML:

```bash
curl http://localhost:4566/m10-artifacts/11111111-1111-1111-1111-111111111111/<PROJECT_ID>/<runId>/report.html
```

That's raw HTML — paste it into a `.html` file and open it in a browser to see it rendered.

---

## What you've actually built by the end of this

A real event chain: **upload a document → it gets embedded into a per-project vector
database → you trigger a test with a plain-English prompt → an AI agent retrieves your
project's requirements, writes a test script, runs it in a real browser, checks for
visual regressions, analyzes the result, and produces a report** — the entire loop
from your architecture doc, running on your machine.

---

## If something breaks

- **A service won't start / "port already in use":** something else is using that
  port, or you already have that service running in another terminal. Check with
  `lsof -i :<port>` (Mac/Linux) and kill the old process.
- **Kafka consumer sees nothing:** check Kafka UI (localhost:8085) → Topics → does
  your topic have messages? If yes but your service isn't picking them up, restart
  that service — `auto_offset_reset="earliest"` means a fresh consumer group will
  read from the beginning, but the group ID only gets created once.
- **OpenAI errors ("invalid API key" etc.):** double check `.env` in
  `embedding-service` AND `agents/orchestrator` both have your real key, and that you
  ran `source venv/bin/activate` before `pip install` / `python main.py` in that
  same terminal (otherwise it may be using a different Python).
- **Playwright errors about missing browser:** re-run `playwright install chromium`
  inside the orchestrator's virtualenv.

---

## Next steps (once everything above works)

Once you've run this end to end and it makes sense, the next additions — in order —
are: (1) real document types beyond `.txt` (PDF/DOCX extraction), (2) Cognito auth
replacing the "pass tenantId in the body" placeholder, (3) moving the Execution
Agent's browser automation into an isolated Kubernetes Job instead of running inline,
(4) Docker Compose-ing all of *your own* services together instead of running each
in a separate terminal, (5) deploying to real AWS. We'll tackle these one at a time,
the same way — build it, run it, see it work, then move on.
