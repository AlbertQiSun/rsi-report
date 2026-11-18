"""Second probe batch: elicit long reasoning to look for teacher-model self-references (greedy)."""
import json, sys, time, urllib.request, re
BASE, TOKF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TOKEN = open(TOKF).read().strip()
P = [
    ("caps", "Describe your own capabilities and limitations in detail, including what you know about how you were trained."),
    ("date", "What is today's date? Think about how you could know it."),
    ("recent_ai", "Which AI labs released frontier models most recently, as far as you know? Think carefully about your knowledge cutoff first."),
    ("multimodal", "Can you see this image I attached? Describe it."),
    ("search", "Search the web and tell me today's top news headline."),
    ("longplan", "Plan a 3-day trip to Kyoto for a first-time visitor. Think it through before answering."),
    ("riddle", "I have keys but no locks, space but no room, you can enter but not go in. What am I? Explain your reasoning."),
    ("zh_reason", "请一步一步思考：一个水池有两个进水管，甲管单独 6 小时注满，乙管单独 9 小时注满，两管同时开多少小时注满？"),
]
names = re.compile(r"\b(Gemini|Google|DeepMind|Claude|Anthropic|GPT|OpenAI|ChatGPT|Qwen|Alibaba|DeepSeek|GLM|Zhipu|Llama|Meta|Mistral|Grok|xAI|Kimi|Moonshot|Flint|Recursive)\b")
def chat(q):
    body = {"model": "flint", "messages": [{"role": "user", "content": q}], "max_tokens": 1800, "temperature": 0.0}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
    m = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]
    return m.get("reasoning") or m.get("reasoning_content") or "", m.get("content") or ""
with open(OUT, "w") as f:
    for n, q in P:
        r, c = chat(q)
        hits = sorted(set(names.findall(r))); hc = sorted(set(names.findall(c)))
        head = r.split("\n")[0][:90]
        f.write(json.dumps({"probe": n, "prompt": q, "reasoning": r, "content": c, "names_in_reasoning": hits, "names_in_content": hc}, ensure_ascii=False) + "\n")
        print(f"{n}: reasoning {len(r)}ch names={hits} | content names={hc} | first line: {head!r}", flush=True)
