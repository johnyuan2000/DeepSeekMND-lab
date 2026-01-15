from fastapi import FastAPI, Request
import requests

app = FastAPI()

LLAMA_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You must respond ONLY in Japanese. Chinese output is strictly forbidden."
}

@app.post("/v1/chat/completions")
async def proxy(req: Request):
    body = await req.json()
    messages = body.get("messages", [])

    # system を強制注入（上書き）
    if not messages or messages[0]["role"] != "system":
        messages = [SYSTEM_PROMPT] + messages
    else:
        messages[0] = SYSTEM_PROMPT

    body["messages"] = messages

    r = requests.post(LLAMA_ENDPOINT, json=body)
    return r.json()
