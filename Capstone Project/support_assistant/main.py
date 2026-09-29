from fastapi import FastAPI
from schemas import AskRequest, AskResponse
from graph import run_query

app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_query(request.query)


@app.get("/health")
def health():
    return {"status": "ok"}
