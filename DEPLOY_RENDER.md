# Deploying Pramāṇa on Render

This guide deploys the full Pramāṇa application as a **single Render Web Service** that serves
both the FastAPI backend and the React frontend from one process.

## Architecture

```
Browser → Render HTTPS → Python (FastAPI / Uvicorn)
                          ├── /api/*     → API routes
                          ├── /files/*   → Course material (PDFs, slide renders)
                          ├── /healthz   → Health check
                          └── /*         → React SPA (from frontend/dist/)
```

- The frontend is built during Render's **build phase** (`npm install && npm run build`).
- The backend serves the built files directly — no separate static site needed.
- API calls use relative URLs (`/api/…`) — no CORS issues.

---

## Prerequisites

- A GitHub account with this repository pushed (public or connected to Render).
- A [Render](https://render.com) account.
- A Google AI Studio API key (for LLM features) — [get one here](https://aistudio.google.com/apikey).
  Without it, Library, search, and Evaluation still work; chat, question generation, and the
  concept graph require a key.

---

## Step-by-step deployment

### 1. Push your code to GitHub

Make sure all changes are committed and pushed:

```bash
git add -A
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Create a Render Web Service

**Option A — Blueprint (recommended):**
1. Go to [Render Dashboard](https://dashboard.render.com) → **Blueprints** → **New Blueprint Instance**.
2. Connect your GitHub repository.
3. Render reads `render.yaml` and creates the service automatically.
4. Set `GEMINI_API_KEY` in the environment variable prompt.

**Option B — Manual setup:**
1. Go to **Dashboard** → **New** → **Web Service**.
2. Connect your GitHub repository.
3. Configure:

| Setting | Value |
|---|---|
| **Name** | `pramana` |
| **Runtime** | Python |
| **Build command** | `pip install -r requirements.txt && cd frontend && npm install && npm run build` |
| **Start command** | `cd backend && python -m pramana.cli serve` |
| **Health check path** | `/healthz` |

4. Under **Environment**, add:

| Key | Value | Secret? |
|---|---|---|
| `PYTHON_VERSION` | `3.11.12` | No |
| `NODE_VERSION` | `20.18.0` | No |
| `PRAMANA_DATA` | `/data` (if using persistent disk) or leave unset (ephemeral) | No |
| `GEMINI_API_KEY` | Your API key | **Yes** |
| `PRAMANA_GENERATOR` | `google:gemini-3.8-flash` | No |
| `PRAMANA_VERIFIER` | `google:gemini-3.7-flash` | No |

### 3. Configure persistent storage (recommended)

> **Free tier limitation:** Render's free tier uses an ephemeral filesystem. All data
> (database, uploads, cached files) is lost on every deploy or restart.
>
> **Starter plan ($7/month)** allows a persistent disk. This is strongly recommended for any
> non-demo use.

If using the Starter plan or above:
1. Go to your service → **Disks** → **Add Disk**.
2. Set **Name** = `pramana-data`, **Mount Path** = `/data`, **Size** = 1 GB.
3. Ensure `PRAMANA_DATA` is set to `/data`.

The SQLite database, downloaded PDFs, slide renders, LLM cache, and embedding model cache are
all stored here and survive restarts.

### 4. Deploy

Click **Deploy** (or it starts automatically). The build takes 3–5 minutes:
1. Python dependencies install.
2. Node dependencies install and Vite builds the frontend.
3. The service starts and passes the `/healthz` check.

### 5. Run course ingestion

After the first deploy, the knowledge base is empty. You must ingest the default course:

**Option A — From the web UI:**
1. Open your Render URL (e.g. `https://pramana-xxxx.onrender.com`).
2. Go to the **Library** page.
3. Click **Ingest default course**. This runs in the background and shows a live log.

**Option B — From Render Shell:**
1. Go to your service → **Shell**.
2. Run:
   ```bash
   cd backend && python -m pramana.cli ingest
   ```
   This fetches MIT OCW lectures and the Grinstead & Snell textbook (~10–15 minutes),
   parses PDFs, downloads YouTube captions, embeds all units, and (with an LLM key) builds
   the concept graph.

> **Important:** If using ephemeral storage (free tier), you need to re-run ingestion after
> every deploy. The web UI approach is more convenient for this.

### 6. Verify deployment

Open your Render URL and confirm:

- [ ] The app loads (React SPA with sidebar navigation)
- [ ] `/healthz` returns `{"status": "ok"}`
- [ ] `/api/status` returns model and KB info
- [ ] Library page shows sources (after ingestion)
- [ ] Search works: try `/api/search?q=Bayes`
- [ ] Tutor page loads (needs a student profile first)
- [ ] Practice page loads
- [ ] Dashboard page loads
- [ ] Evaluation page shows pre-computed results

---

## Environment variables reference

| Variable | Required? | Default | Description |
|---|---|---|---|
| `PORT` | Auto-set by Render | `8000` | HTTP listen port |
| `PYTHON_VERSION` | Recommended | — | Python version for Render |
| `NODE_VERSION` | Recommended | — | Node.js version for Render |
| `PRAMANA_DATA` | If persistent disk | `<project>/data` | Root path for all data |
| `GEMINI_API_KEY` | For LLM features | — | Google AI Studio key |
| `ANTHROPIC_API_KEY` | Optional | — | Anthropic key (alternative provider) |
| `PRAMANA_GENERATOR` | No | `google:gemini-3.8-flash` | Generator model |
| `PRAMANA_VERIFIER` | No | `google:gemini-3.7-flash` | Verifier model |
| `PRAMANA_BULK` | No | Same as generator | Bulk worker model |
| `PRAMANA_EMBED_MODEL` | No | `BAAI/bge-small-en-v1.5` | Embedding model |
| `PRAMANA_COVERAGE_THRESHOLD` | No | `0.72` | Search similarity gate |
| `PRAMANA_MAX_FIGURE_PAGES` | No | `60` | Max textbook pages for vision |

---

## Persistent data inventory

Everything below lives under `PRAMANA_DATA` (default: `data/`):

| Path | Contents | Recreatable? |
|---|---|---|
| `pramana.sqlite` | Full database: KB, students, mastery, questions, chats, jobs | Ingestion recreates KB; student data is **lost** |
| `raw/` | Downloaded PDFs, caption caches, uploaded files | Re-ingestion re-downloads; uploads are **lost** |
| `renders/` | Slide and page PNG images | Re-ingestion recreates |
| `cache/` | LLM response cache | Saves money; not required |
| `cache/models/` | fastembed model files (~130 MB) | Auto-downloaded on first use |

---

## Known limitations

1. **Free tier = ephemeral storage.** Student progress, uploaded documents, chat history, and
   ingested data are lost on every deploy or restart. Use the Starter plan with a persistent
   disk for any serious use.

2. **Ingestion time.** Fetching 8 lectures + textbook takes ~10–15 minutes. YouTube may
   rate-limit caption requests. The ingestion is idempotent and cached, so re-runs are faster.

3. **Memory.** The embedding model (~130 MB) plus the in-memory index fit within Render's
   512 MB free tier. Larger courses may need more memory.

4. **Render free tier spin-down.** Free services spin down after 15 minutes of inactivity.
   First request after spin-down takes ~30 seconds and re-initialises the embedding model.

5. **Concurrent writes.** SQLite with a single-threaded lock (as implemented) is safe for
   one web service. Do not scale to multiple instances without switching to PostgreSQL.

6. **No automatic ingestion.** Ingestion is deliberately not run on startup to avoid blocking
   the web server or making expensive LLM calls on every deploy.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Health check fails | Build error or missing dependency | Check Render build logs |
| "No concept graph" error | Ingestion hasn't run | Run ingestion from Library page or Shell |
| 503 on chat/questions | No API key configured | Set `GEMINI_API_KEY` in environment |
| Slides not showing | `PRAMANA_DATA` not pointing to persistent disk | Check disk mount path |
| Data lost after deploy | No persistent disk | Add a disk or accept ephemeral mode |
