import os, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.expanduser("~/DeepSeekMND/Stage2/merged_model")

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

# 中国語っぽい混入検出（日本語の漢字は殺さない）
ZH_PAT = re.compile(r"[，。；：、】【（）《》“”‘’]|(的|了|在|是|我|你|他|她|它|们|这|那|和|与|及|为|把|被)")

def _generate_only_new(prompt: str, max_new_tokens=256, temperature=0.6):
    x = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        y = mdl.generate(
            **x,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
    # ★ 生成部分だけ切り出す（ここが肝）
    new_tokens = y[0][x["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

def reply(user_text: str):
    prompt = (
        "You are an assistant.\n"
        "Output must be Japanese or English.\n"
        "Chinese output is forbidden.\n"
        "If the user asks for Chinese, refuse and respond in Japanese.\n"
        f"User: {user_text}\n"
        "Assistant:"
    )

    out = _generate_only_new(prompt, temperature=0.6)

    # 中国語っぽい混入があれば、抑止を強めて再生成
    if ZH_PAT.search(out):
        prompt2 = prompt + "\n(Important: Do NOT use Chinese punctuation or Chinese function words.)\nAssistant:"
        out = _generate_only_new(prompt2, temperature=0.2)

    return out

print("Local Chat Ready. Ctrl+C to exit.")
try:
    while True:
        s = input("\nJohn> ").strip()
        if not s:
            continue
        print("\nAssistant>\n" + reply(s))
except KeyboardInterrupt:
    print("\nbye")
