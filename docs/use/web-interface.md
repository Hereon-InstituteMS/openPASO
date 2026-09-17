# Web interface

!!! warning "Experimental"
    A browser interface for openPASO is being built right now. It changes often and is not one of
    the two supported ways above. If you want something that works today, use
    [Option A](ai-app.md) or [Option B](api-key.md).

The interface lets you type a request, watch the model's tool calls as they happen, browse the files
a run produced, and see plots of the results. It can use your OpenRouter key (the same `.env` as
Option B), a local model server, or Claude Code.

To try it, from the openPASO folder with the agent packages installed:

```bash
pip install fastapi 'uvicorn[standard]' python-multipart websockets
uvicorn webui.app:app --port 8080
# then open http://localhost:8080
```

Details and current state: `webui/README.md` in the repository.
