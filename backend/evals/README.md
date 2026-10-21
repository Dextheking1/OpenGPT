# LangSmith trace-based evaluation

This folder contains a lightweight evaluation scaffold that reads the traces
already emitted by OpenGPT into LangSmith and produces a JSON report.

## What it checks

- `chat_node` / `My GPT` style traces:
  - answer is present
  - the trace is error-free
  - the response has some substance
- document RAG traces:
  - retrieved context is present
  - source file metadata is present
- document ingest traces:
  - the document was indexed into chunks successfully

## Run it

```bash
cd backend
python -m evals.online_eval --project OpenGPT --limit 50
```

The report is written to `backend/evals/results/`.

## Notes

- Make sure `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=OpenGPT` are set.
- The app now loads LangSmith config before importing the traced modules, so the
  traced nodes are picked up consistently when the backend starts.
- If you later add `@traceable(...)` to more service methods, they will show up
  in the same report automatically as long as their names are listed in
  `backend/evals/online_eval.py`.
