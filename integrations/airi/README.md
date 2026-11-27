# Connect Project AIRI to Hmph

AIRI provides the character, voice, and interface. Hmph adds relevant memories to the request and forwards the upstream reply as a stream.

## Start the gateway

Follow the [installation and model setup](../../README.md#quickstart-for-project-airi), then run:

```bash
hmph airi-gateway --db data/hmph.db --port 8018
```

## Connect AIRI

In **Settings → Providers → OpenAI Compatible**, use these values.

- **Base URL** `http://127.0.0.1:8018/v1`
- **API Key** `local-hmph` for an unauthenticated local gateway. If you set `HMPH_GATEWAY_API_KEY` in the gateway process environment, enter that key instead.
- **Model** `hmph` uses the default memory namespace and the configured upstream model. Use `hmph/yuki` for a separate character's memory, or `hmph/yuki@<upstream_model>` to choose a different model on the same upstream service.

To check the integration, record a piano practice session for Saturday, move it to Monday, then ask for the current day in a new chat using the same model route. You can inspect stored changes at `GET /v1/memory` and the assembled context at `POST /v1/memory/recall`.

### Optional request headers

- `X-Hmph-Namespace` selects the memory namespace and overrides the model route.
- `X-Hmph-Session-Id` ties requests to a specific conversation.
- `X-Hmph-Memory-Mode` selects `auto` for recall and writes, `readonly` for recall only, or `bypass` for upstream chat only.

## Connect an MCP client

Use [mcp-config.json](mcp-config.json) as an example. The stdio command is `hmph local-mcp --db data/hmph.db`. When the gateway is running, Streamable HTTP clients can connect to `http://127.0.0.1:8018/mcp`.

## Optional client integrations

- Subscribe to `GET /v1/memory/events?namespace=yuki` for memory update events over SSE.
- Call `GET /v1/companion/proactive?namespace=yuki&cooldown_hours=12` from your own idle timer. The API offers suggestions based on unresolved references and active plans. Paused plans are excluded from the active-plan check.
- Manage core memory with `GET /v1/memory/core` and `PUT` or `DELETE /v1/memory/core/{block_key}`.
- Trigger a session summary with `POST /v1/memory/summarize`. For clients using the default session route, Hmph starts a new episode after the configured idle gap.
- Use `GET /v1/memory/as-of` to look up state by effective date. JSON and Markdown snapshots are available via `GET /v1/memory/export`.
- Use `GET /v1/memory/extractions` to inspect quarantined messages, and `POST /v1/memory/extractions/{message_id}/retry` to retry one.

