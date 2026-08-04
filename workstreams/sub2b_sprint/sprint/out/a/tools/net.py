"""Pooled HTTP layer for header reads. Reuses connections; resolves the HF
redirect to the CDN once per file so subsequent Range reads skip the 302."""
import json
import os
import struct
import time

import requests

HF = "https://huggingface.co"
S = requests.Session()
S.headers["User-Agent"] = "lane-a-ledger/1.0 (header-range-reader)"
if os.environ.get("HF_TOKEN"):
    S.headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]

_CDN = {}


def _retry(fn, n=4):
    last = None
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def cdn(url):
    """Follow the HF 302 once and cache the signed CDN url."""
    if url in _CDN and _CDN[url][1] > time.time():
        return _CDN[url][0]

    def go():
        r = S.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True,
                  timeout=60, stream=True)
        r.close()
        return r

    r = _retry(go)
    final = r.url
    _CDN[url] = (final, time.time() + 900)
    return final


def rng(url, start, end, direct=False):
    """Range read. Signed CDN URLs expire and sometimes 403; on any 4xx we drop
    the cached signature and go back through the HF redirect."""
    last = None
    for attempt in range(6):
        try:
            r = S.get(url, headers={"Range": "bytes=%d-%d" % (start, end)},
                      timeout=180, allow_redirects=True)
            if r.status_code in (200, 206):
                got = r.content
                if r.status_code == 200 and (end - start + 1) < len(got):
                    got = got[start:end + 1]  # server ignored Range
                return got
            raise IOError("%s on %s" % (r.status_code, url[:90]))
        except Exception as e:
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise last


def get(url, **kw):
    return _retry(lambda: S.get(url, timeout=120, **kw))


def api(path):
    r = get(HF + path)
    r.raise_for_status()
    return r.json()


def resolve(repo, path, rev="main", kind="models"):
    if kind == "models":
        return "%s/%s/resolve/%s/%s" % (HF, repo, rev, path)
    return "%s/%s/%s/resolve/%s/%s" % (HF, kind.rstrip("s"), repo, rev, path)


def tree(repo, rev="main", kind="models", path=""):
    out = []
    url = HF + "/api/%s/%s/tree/%s/%s?recursive=true&expand=true" % (
        kind, repo, rev, path)
    while url:
        r = get(url)
        r.raise_for_status()
        out.extend(r.json())
        link = r.headers.get("Link", "")
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    return out


def sizeof(entry):
    return entry.get("size") or (entry.get("lfs") or {}).get("size") or 0


def st_header(url):
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    if n <= 0 or n > 200_000_000:
        raise ValueError("implausible header len %d" % n)
    raw = rng(url, 8, 8 + n - 1)
    return json.loads(raw.decode("utf-8")), n


def st(repo, path, rev="main", kind="models"):
    u = resolve(repo, path, rev, kind)
    h, n = st_header(u)
    meta = h.pop("__metadata__", None)
    return h, meta, n, u
