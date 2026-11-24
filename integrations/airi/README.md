# Connecting Project AIRI to Hmph Memory Component (v4.0)

Hmph operates as a **headless, single-file SQLite frontier memory component** — zero Web UI, zero Docker, zero AIRI fork required. AIRI handles Live2D/VRM visuals, STT/TTS voice, emotion tags, and desktop stage; Hmph handles bi-temporal state evolution, multi-character isolation, vector-indexed sleep consolidation, paused-plan anti-nagging, and atomic zero-resurrection forgetting.

---

## Step 1: Start Hmph Headless AIRI Gateway

Ensure `.env.local` (or environment variables) contains your upstream LLM key (`HMPH_DEEPSEEK_API_KEY` or `OPENAI_API_KEY`, plus optional `HMPH_LOCAL_BASE_URL` / `HMPH_LOCAL_MODEL`):

```powershell
hmph airi-gateway --db artifacts/local-memory/hmph.db --port 8018
```

Endpoints exposed locally:
- **OpenAI-Compatible Base URL** (CORS enabled for AIRI Web & Tauri Desktop): `http://127.0.0.1:8018/v1`
- **MCP Streamable HTTP JSON-RPC**: `http://127.0.0.1:8018/mcp`
- **Headless Memory REST API**: `http://127.0.0.1:8018/v1/memory`

---

## Step 2: Connect AIRI Frontend (Web or Desktop)

1. Open **AIRI** → **Settings** → **Providers** → **OpenAI Compatible**.
2. Fill in:
   - **Base URL**: `http://127.0.0.1:8018/v1`
   - **API Key**: `local-hmph` (any non-empty string; real upstream key stays safely on your local Hmph daemon)
   - **Model**:
     - `hmph` — default character namespace (`default`), routed to your configured upstream model.
     - `hmph/<character_name>` (e.g., `hmph/yuki`, `hmph/airi`) — **strictly isolated per-character memory namespace** inside the same SQLite file!
     - `hmph/<character_name>@<upstream_model>` (e.g., `hmph/yuki@deepseek-chat`) — override both character namespace and upstream model per AIRI profile.
3. Chat normally in AIRI (voice or text):
   - AIRI's own **character prompt, Live2D emotion tags, and TTS pacing rules are 100% preserved**.
   - Hmph transparently injects `<hmph_memory>` context before forwarding to the upstream LLM and streams OpenAI SSE chunks back immediately (zero blocking of AIRI's voice/lip-sync loop).
   - State-evolution extraction (`create / reschedule / pause / resume / cancel / retract / ambiguous`) and Sleep Consolidation run asynchronously in the background.

### Optional Request Headers
If you call the gateway from a custom AIRI plugin or middleware:
- `X-Hmph-Namespace`: explicitly select character namespace (overrides model suffix).
- `X-Hmph-Session-Id`: bind conversation turns to a specific session ID.
- `X-Hmph-Memory-Mode`: `auto` (default: recall + write + extract) | `readonly` (recall only) | `bypass` (pure LLM pass-through).

---

## Step 3 (Optional): Mount Hmph MCP Tools in AIRI Desktop (Tauri MCP)

Use `integrations/airi/mcp-config.json` to expose all **8 explicit memory tools** (`hmph_remember`, `hmph_recall`, `hmph_list_memories`, `hmph_forget`, `hmph_consolidate`, `hmph_graph`, `hmph_toggle_pause`, `hmph_core_memory`) directly to AIRI Desktop or any MCP client (Claude Desktop, Cursor, etc.).

- **Stdio mode**: launches `python -m hmph.cli local-mcp --db artifacts/local-memory/hmph.db`
- **HTTP mode** (when `hmph airi-gateway` is already running): send MCP JSON-RPC 2.0 requests directly to `POST http://127.0.0.1:8018/mcp`.

---

## Step 4 (v5.0 Deep Companion Cooperation): Core Memory, Live Event Bus & Proactive Check-In

Hmph v5.0 goes beyond passive proxying with three dedicated AIRI companion cooperation endpoints:

1. **Always-in-Context Core Memory (`GET/PUT/DELETE /v1/memory/core`)**:
   - Auto-populated during Sleep Consolidation (and editable via REST or `hmph_core_memory` MCP tool).
   - Injected into every AIRI turn (even casual greetings) while strictly respecting paused-plan boundaries and cascade-forgetting with zero resurrection.
2. **Real-Time Memory Event Stream (`GET /v1/memory/events?namespace=yuki`)**:
   - Standard SSE stream emitting `memory_extracted`, `item_updated`, `ambiguity_detected`, `consolidation_completed`, `session_summarized`, `core_memory_updated`, and `memory_forgotten`.
   - AIRI Desktop plugins / Live2D hooks can subscribe with `EventSource("http://127.0.0.1:8018/v1/memory/events?namespace=yuki")` to trigger subtle nod/blush reactions when a promise is saved or updated.
3. **Boundary-Safe Proactive Companion Check-In (`GET /v1/companion/proactive?namespace=yuki&cooldown_hours=12`)**:
   - Called on AIRI's idle/heartbeat timer. Strictly excludes `paused`, `cancelled`, `retracted`, and `completed` items, returning `{should_initiate, reason, topic, suggested_cue}` so AIRI only speaks up politely for open ambiguities or due active commitments.
4. **Coreference-Resolved Session Summaries (`POST /v1/memory/summarize`)**:
   - Distills completed sessions into pronoun-resolved, temporally-anchored atomic facts (zero unresolved `"my GF"` / `"them"` / `"last week"` gaps).
