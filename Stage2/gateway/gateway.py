import os
from fastapi import FastAPI, Request
import httpx

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8080")

SYSTEM_ENFORCE = (
    "You are a helpful assistant.\n"
    "Output must be Japanese or English.\n"
    "Chinese output is forbidden.\n"
    "If the user asks for Chinese, refuse and respond in Japanese.\n"
    "Do NOT use Chinese punctuation or Chinese function words.\n"
)

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM}

@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(f"{UPSTREAM}/v1/models")
        return r.json()

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()

    msgs = body.get("messages", [])
    # 先頭に強制systemを注入（既存systemがあっても先頭優先にする）
    enforced = [{"role": "system", "content": SYSTEM_ENFORCE}] + msgs
    body["messages"] = enforced

    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"{UPSTREAM}/v1/chat/completions", json=body)
        return r.json()
