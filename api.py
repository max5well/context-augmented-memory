from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from modules.context_service import get_context

app = FastAPI()


class Request(BaseModel):
    prompt: str
    top_k: int = 5


@app.post("/retrieve-context")
def retrieve_context_api(request: Request):
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    context = get_context(
        prompt=prompt,
        top_k=request.top_k
    )

    return {
        "prompt": prompt,
        "context": context
    }

