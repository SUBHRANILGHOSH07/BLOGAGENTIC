# BlogAgentic

Generate a blog from a topic, with optional Hindi or French translation.

## Start the HTTP API

Run these commands in PowerShell. Keep the server terminal open while sending requests.

```powershell
Set-Location 'C:\Users\indra\OneDrive\Desktop\BLOGAGENTIC'
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The API documentation is available at <http://127.0.0.1:8000/docs>. `main.py`
only prints a greeting; it does not start the server. `ECONNREFUSED
127.0.0.1:8000` means the client could not reach a listener at that address:
start the API and check its terminal for any startup error.

The project uses Python 3.13 or newer. If the local environment needs to be
installed, run `uv sync` from this folder. Configure `GROQ_API_KEY` in `.env`
before startup. `GROQ_MODEL` is optional; the current default is
`openai/gpt-oss-20b`. A LangSmith API key is optional unless tracing is enabled.
Do not commit `.env` or share API keys.

Send `POST /blogs` with a JSON body:

```json
{
  "topic": "How AI agents work",
  "language": "hindi"
}
```

`topic` must be a nonblank string. Omit `language`, use an empty string, or use
`english` for an English blog. `hindi` and `french` request translation.
Language names are normalized for surrounding whitespace and letter case.
Invalid requests return HTTP 422. Successful responses have the form
`{"data": {...}}`.

## Run the graph in LangGraph Studio

The LangGraph development server is separate from the FastAPI server above.

```powershell
.\.venv\Scripts\langgraph.exe dev
```

Use the API and Studio addresses printed by that command. It exposes the
`blog_generator_agent` graph configured in `langgraph.json`; it does not start
the FastAPI `/blogs` route on port 8000.

Studio accepts the graph state field `current_language`, while the HTTP API
accepts `language`:

```json
{
  "topic": "How AI agents work",
  "current_language": "hindi"
}
```

Omitting `current_language` defaults to English. Supported graph language
values are `english`, `hindi`, and `french`.

## Translation output

Translation uses Groq's native strict JSON schema output with the default
`openai/gpt-oss-20b` model. It does not require the model to call a tool.
If changing `GROQ_MODEL`, choose a model that supports Groq's strict structured
outputs: https://console.groq.com/docs/structured-outputs.

Translation streaming is explicitly disabled for compatibility with this mode,
including in Studio. Responses that hit the output token limit are rejected
instead of returning a cut-off blog. Title and content generation can still stream.

Run the offline regression tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
