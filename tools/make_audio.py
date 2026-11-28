"""Makes the app's audio with the Kokoro voice "Heart" (af_heart, Apache-2.0, runs on CPU).

Reads the app's own data (checked out to ./app) and writes MP3 files next to this script's repository:
  w/<key>.mp3          words, slang, useful expressions and their example sentences (kept for good)
  a/<article>/<key>.mp3  every sentence of every article of the last 30 days (older ones are removed)
  index.json           which words and which articles have audio (the app reads it)
<key> = first 12 hex chars of sha1(text.strip(), spaces collapsed, lower case) - the app computes the same.
Only what is missing is made; a run stops after BUDGET_MIN minutes and the next run continues.
"""
import hashlib, json, os, re, shutil, time
from datetime import datetime, timedelta, timezone

import numpy as np
import lameenc
from kokoro import KPipeline

APP = os.environ.get("APP_DIR", "app")
BUDGET = float(os.environ.get("BUDGET_MIN", "150")) * 60
KEEP_DAYS = 30
VOICE, SPEED = "af_heart", 0.95
start = time.time()


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip()).lower()


def key(t):
    return hashlib.sha1(norm(t).encode("utf-8")).hexdigest()[:12]


def split_sentences(text):  # the same split as the app (js/app.js splitSentences)
    return re.findall(r'[^.!?]+(?:[.!?]+["”’)\]]*\s*|$)', text) or [text]


pipe = KPipeline(lang_code="a")


def mp3(text, path):
    chunks = [a.numpy() if hasattr(a, "numpy") else a for _, _, a in pipe(text, voice=VOICE, speed=SPEED)]
    if not chunks:
        return False
    pcm = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(32); enc.set_in_sample_rate(24000); enc.set_channels(1); enc.set_quality(2)
    data = enc.encode(pcm.tobytes()) + enc.flush()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return True


def out_of_time():
    return time.time() - start > BUDGET


# ---------- what needs audio ----------
vocab = json.load(open(f"{APP}/data/vocab.json", encoding="utf-8"))
data_js = open(f"{APP}/js/data.js", encoding="utf-8").read()
word_texts = []  # (priority, text): newest first
for w in vocab["words"]:
    for t in (w[0], w[4]):
        if t: word_texts.append((w[6] or "", t))
for e in vocab.get("exprs", []):
    if e[0] in ("slang", "phrase"):
        for t in (e[1], e[4]):
            if t: word_texts.append(("9999", t))  # slang and expressions first
for block in re.findall(r"^[a-z]+: `([\s\S]*?)`", data_js, re.M):  # built-in words: word|he|pos|lvl|emoji|ipa|example|...
    for line in block.split("\n"):
        p = line.split("|")
        if len(p) > 6:
            for t in (p[0], p[6]):
                if t.strip(): word_texts.append(("", t.strip()))
for line in data_js.split("\n"):  # built-in slang: slang|cat|term|he|lit|real|example|...
    p = line.split("|")
    if len(p) > 7 and p[0] == "slang":
        for t in (p[2], p[6]):
            if t.strip(): word_texts.append(("9999", t.strip()))
# example sentences written later for words that had none (data/meanings.json "ex")
if os.path.exists(f"{APP}/data/meanings.json"):
    for v in json.load(open(f"{APP}/data/meanings.json", encoding="utf-8")).get("words", {}).values():
        if v.get("ex"): word_texts.append(("9998", v["ex"]))
word_texts.sort(key=lambda x: x[0], reverse=True)

news = json.load(open(f"{APP}/data/news/index.json", encoding="utf-8"))
cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)
recent = [a for a in news.get("articles", []) if datetime.fromisoformat(a["first_published_in_app_at"].replace("Z", "+00:00")) >= cutoff]
recent.sort(key=lambda a: a["first_published_in_app_at"], reverse=True)
# articles the reader deleted with the trash button: no audio, and their audio is removed
hidden = set(json.load(open(f"{APP}/data/hidden.json"))) if os.path.exists(f"{APP}/data/hidden.json") else set()
recent = [a for a in recent if a["id"] not in hidden]

index = json.load(open("index.json")) if os.path.exists("index.json") else {"words": "", "articles": {}}
have_words = set(filter(None, index.get("words", "").split(",")))
done_articles = dict(index.get("articles", {}))
made = 0
import subprocess
PUBLISH_EVERY = 30 * 60
last_publish = time.time()


def write_index():
    idx = {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "words": ",".join(sorted(have_words)), "articles": done_articles}
    json.dump(idx, open("index.json", "w"), separators=(",", ":"))


def maybe_publish(force=False):
    """Publish what is ready every 30 minutes, so the app gets new audio during a long run."""
    global last_publish
    if not force and time.time() - last_publish < PUBLISH_EVERY:
        return
    last_publish = time.time()
    write_index()
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    run = lambda *a: subprocess.run(list(a), check=False)
    run("git", "add", "-A")
    run("git", "commit", "-q", "-m", "Audio (in progress)")
    run("git", "push", "-q", "-f", "origin", "HEAD:main")
    run("gh", "api", "-X", "POST", f"repos/{os.environ.get('GITHUB_REPOSITORY')}/pages/builds")
    print("published", len(have_words), "words,", len(done_articles), "articles", flush=True)



# ---------- words, slang, expressions and their examples (kept for good) ----------
def do_words(items):
    global made
    for _, t in items:
        if out_of_time(): break
        k = key(t)
        if k in have_words: continue
        f = f"w/{k}.mp3"
        if os.path.exists(f) or mp3(t, f):
            have_words.add(k); made += 1
        maybe_publish()


# first: slang, expressions and the words of the last 7 days
recent_cut = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
do_words([x for x in word_texts if x[0] >= recent_cut])

# ---------- articles of the last 30 days (complete ones are listed in the index) ----------
keep_ids = {a["id"] for a in recent}
for a in recent:
    if out_of_time(): break
    if a["id"] in done_articles: continue
    path = f"{APP}/data/news/{a['id']}.json"
    if not os.path.exists(path): continue
    body = json.load(open(path, encoding="utf-8"))
    sentences = [s for p in body.get("paragraphs", []) for s in split_sentences(p) if norm(s)]
    ok = True
    for s in sentences:
        if out_of_time(): ok = False; break
        f = f"a/{a['id']}/{key(s)}.mp3"
        if not os.path.exists(f):
            mp3(s, f); made += 1
        maybe_publish()
    if ok: done_articles[a["id"]] = len(sentences)
# older articles go (the app keeps favourites' audio on the device)
for d in os.listdir("a") if os.path.isdir("a") else []:
    if d not in keep_ids:
        shutil.rmtree(f"a/{d}", ignore_errors=True); done_articles.pop(d, None)

# ---------- all the remaining words (kept for good) ----------
do_words(word_texts)

# ---------- index + size check ----------
write_index()
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(".") if ".git" not in r and "app" not in r.split(os.sep)[:2] for f in fs)
total_words = len({key(t) for _, t in word_texts})
print(f"made {made} clips in {int((time.time() - start) / 60)} min · words {len(have_words)}/{total_words} · articles {len(done_articles)}/{len(recent)} · size {size / 1e6:.0f} MB")
if size > 800e6:
    print("::warning::Audio storage is above 800 MB (GitHub Pages limit is 1 GB). Time to add a second audio repository.")
