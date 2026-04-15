# Layer-zero (L0) architecture

Layers and data flow inside the `reviewagent` process. Usage: [README.md](../README.md). Tool packs: [agent-tool-packs.md](./agent-tool-packs.md).

```
+------------------------------------------------------------------+
|  Ingress                                                         |
|  +-----------+   +----------------+   +-----------+               |
|  | HTTP API  |   | CLI (+ batch)  |   | TUI       |               |
|  | (FastAPI) |   | reviewagent.cli|   | embedded  |               |
|  +-----------+   +----------------+   +-----------+               |
|        \                |                /                        |
|         +---------------+---------------+                         |
+------------------------------------------------------------------+
                          |
                          v
+------------------------------------------------------------------+
|  Service layer (reviewagent.api)                                 |
|  Routing & validation · limits · sync/async review · audit trail |
+------------------------------------------------------------------+
                          |
                          v
+------------------------------------------------------------------+
|  Orchestration (reviewagent.agent · ReviewOrchestrator)          |
|  Pipeline short-circuit → LLM tool loop (multi-turn) · opt. RAG  |
+------------------------------------------------------------------+
                          |
                          v
+------------------------------------------------------------------+
|  Review engine                                                   |
|  +----------------+  +----------------+  +----------------+       |
|  | Recall         |  | Detector       |  | Model          |       |
|  | pipeline/      |  | toolpacks/     |  | adapters/      |       |
|  | wordlist·hash  |  | text·img·a/v   |  | LLM factory    |       |
|  +----------------+  +----------------+  +----------------+       |
+------------------------------------------------------------------+
                          |
                          v
+------------------------------------------------------------------+
|  Storage & observability                                         |
|  SQLite audit · async queue · session memory · logs / metrics    |
+------------------------------------------------------------------+
```

## Notes

- **External LLM / embeddings**: remote calls from `adapters/` and `rag/`; not persisted in-process storage.
- **Shared orchestration**: CLI, file loads, and HTTP all converge on the same `ReviewOrchestrator`.
- This diagram is the **single-node default**; multi-instance deployments should replace SQLite with distributed storage.
