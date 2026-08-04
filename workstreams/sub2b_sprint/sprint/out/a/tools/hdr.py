"""Lane A header-parsing toolkit.

Everything here reads *headers* over HTTP Range. No model weights are downloaded
except where a file is small enough to be enumerated locally (ONNX initializers).

Evidence classes produced:
  EXACT  - tensor-level parse (shape product summed, dtype known)
  UB     - bound from bytes / archive size / vendor round figure
  UNKNOWN- no basis

Usage: import as a module, or run subcommands.
"""
import json
import os
import struct
import sys
import time
import urllib.request
import urllib.error

HF = "https://huggingface.co"
UA = {"User-Agent": "lane-a-ledger/1.0 (header-range-reader)"}
TOK = os.environ.get("HF_TOKEN")

# safetensors dtype -> bytes per element
DT = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
# dtypes that carry no learned value (index/shape buffers)
NON_LEARNED = {"I64", "U64", "I32", "U32", "I16", "U16", "I8", "U8", "BOOL"}


def _req(url, headers=None, retries=4):
    h = dict(UA)
    if TOK:
        h["Authorization"] = "Bearer " + TOK
    if headers:
        h.update(headers)
    last = None
    for i in range(retries):
        try:
            r = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(r, timeout=90) as resp:
                return resp.read(), dict(resp.headers), resp.geturl()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                last = e
                time.sleep(2 * (i + 1))
                continue
            raise
        except Exception as e:  # transient network
            last = e
            time.sleep(2 * (i + 1))
    raise last


def rng(url, start, end):
    """Inclusive byte range read."""
    b, h, _ = _req(url, {"Range": "bytes=%d-%d" % (start, end)})
    return b


def resolve(repo, path, rev="main"):
    return "%s/%s/resolve/%s/%s" % (HF, repo, rev, path)


def head(url):
    h = dict(UA)
    if TOK:
        h["Authorization"] = "Bearer " + TOK
    r = urllib.request.Request(url, headers=h, method="HEAD")
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return dict(resp.headers), resp.status
    except urllib.error.HTTPError as e:
        return dict(e.headers), e.code


def api(path):
    b, _, _ = _req(HF + path)
    return json.loads(b)


def repo_files(repo, rev="main", kind="models"):
    """Full recursive file tree (the HF API 'siblings' list is flat but complete)."""
    out = []
    d = api("/api/%s/%s/revision/%s" % (kind, repo, rev))
    for s in d.get("siblings", []):
        out.append(s["rfilename"])
    return out, d


def tree(repo, rev="main", kind="models"):
    """Recursive tree with sizes (paginated)."""
    out = []
    url = HF + "/api/%s/%s/tree/%s?recursive=true&expand=true" % (kind, repo, rev)
    while url:
        b, h, _ = _req(url)
        page = json.loads(b)
        out.extend(page)
        link = h.get("Link", "") or h.get("link", "")
        url = None
        if 'rel="next"' in link:
            url = link.split(";")[0].strip("<> ")
    return out


# ---------------------------------------------------------------- safetensors

def st_header(url):
    """Return (header_dict, header_len, file_size_hint)."""
    first8 = rng(url, 0, 7)
    n = struct.unpack("<Q", first8)[0]
    if n <= 0 or n > 200_000_000:
        raise ValueError("implausible safetensors header length %d" % n)
    raw = rng(url, 8, 8 + n - 1)
    return json.loads(raw.decode("utf-8")), n


def st_tensors(repo, path, rev="main"):
    url = resolve(repo, path, rev)
    hdr, n = st_header(url)
    meta = hdr.pop("__metadata__", None)
    return hdr, meta, n, url


def prod(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def st_summary(hdr):
    """Sum parameters, split learned (float) vs non-learned (int/bool)."""
    learned = 0
    nonlearned = 0
    by_dtype = {}
    for name, t in hdr.items():
        p = prod(t["shape"])
        d = t["dtype"]
        by_dtype[d] = by_dtype.get(d, 0) + p
        if d in NON_LEARNED:
            nonlearned += p
        else:
            learned += p
    return {"learned": learned, "non_learned": nonlearned,
            "total": learned + nonlearned, "by_dtype": by_dtype,
            "n_tensors": len(hdr)}


def st_probe_bytes(url, hdr, header_len, name, offsets_within):
    """Read a few bytes from inside a tensor's data region."""
    base = 8 + header_len
    s, e = hdr[name]["data_offsets"]
    out = []
    for off in offsets_within:
        a = base + s + off
        out.append(rng(url, a, a + 15))
    return out, (s, e)


# ------------------------------------------------------------------ gguf

GGUF_T = {0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2), 4: ("u32", 4),
          5: ("i32", 4), 6: ("f32", 4), 7: ("bool", 1), 8: ("string", 0),
          9: ("array", 0), 10: ("u64", 8), 11: ("i64", 8), 12: ("f64", 8)}


class Buf:
    def __init__(self, url, chunk=1 << 20):
        self.url = url
        self.chunk = chunk
        self.data = b""
        self.base = 0
        self.pos = 0

    def need(self, n):
        while self.pos + n > len(self.data):
            more = rng(self.url, len(self.data), len(self.data) + self.chunk - 1)
            if not more:
                raise EOFError
            self.data += more

    def take(self, n):
        self.need(n)
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def string(self):
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")

    def value(self, t):
        name, sz = GGUF_T[t]
        if name == "string":
            return self.string()
        if name == "array":
            et = self.u32()
            n = self.u64()
            return [self.value(et) for _ in range(n)]
        raw = self.take(sz)
        fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I",
               "i32": "<i", "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q",
               "f64": "<d"}[name]
        return struct.unpack(fmt, raw)[0]


def gguf_header(url):
    b = Buf(url)
    magic = b.take(4)
    if magic != b"GGUF":
        raise ValueError("not gguf: %r" % magic)
    ver = b.u32()
    n_tensors = b.u64()
    n_kv = b.u64()
    kv = {}
    for _ in range(n_kv):
        k = b.string()
        t = b.u32()
        kv[k] = b.value(t)
    tensors = []
    for _ in range(n_tensors):
        nm = b.string()
        nd = b.u32()
        dims = [b.u64() for _ in range(nd)]
        ttype = b.u32()
        off = b.u64()
        tensors.append({"name": nm, "shape": dims, "ggml_type": ttype, "offset": off})
    return {"version": ver, "kv": {k: v for k, v in kv.items()
                                   if not isinstance(v, list) or len(v) < 12},
            "n_tensors": n_tensors, "tensors": tensors}


# ------------------------------------------------------------------ onnx

def onnx_initializers(path):
    """Enumerate initializers of a LOCAL .onnx file without onnxruntime.

    Minimal protobuf walk: ModelProto.graph(field 7) -> GraphProto.initializer(field 5)
    -> TensorProto{dims=1(repeated int64), data_type=2(int32), name=8(string),
       raw_data=9(bytes), external_data=13}
    """
    with open(path, "rb") as f:
        data = f.read()
    return onnx_initializers_bytes(data)


def _pb_iter(buf, start, end):
    i = start
    while i < end:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
            yield fn, wt, v, None
        elif wt == 1:
            yield fn, wt, None, (i, i + 8)
            i += 8
        elif wt == 2:
            n, i = _varint(buf, i)
            yield fn, wt, None, (i, i + n)
            i += n
        elif wt == 5:
            yield fn, wt, None, (i, i + 4)
            i += 4
        else:
            raise ValueError("wiretype %d at %d" % (wt, i))


def _varint(buf, i):
    r = 0
    s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7F) << s
        if not (b & 0x80):
            return r, i
        s += 7


ONNX_DT = {1: ("FLOAT", 4), 2: ("UINT8", 1), 3: ("INT8", 1), 4: ("UINT16", 2),
           5: ("INT16", 2), 6: ("INT32", 4), 7: ("INT64", 8), 9: ("BOOL", 1),
           10: ("FLOAT16", 2), 11: ("DOUBLE", 8), 12: ("UINT32", 4),
           13: ("UINT64", 8), 14: ("COMPLEX64", 8), 15: ("COMPLEX128", 16),
           16: ("BFLOAT16", 2)}


def _tensorproto(buf, s, e):
    dims, dt, name, raw, ext = [], None, "", None, None
    for fn, wt, v, span in _pb_iter(buf, s, e):
        if fn == 1 and wt == 0:
            dims.append(v)
        elif fn == 1 and wt == 2:  # packed dims
            a, b = span
            j = a
            while j < b:
                x, j = _varint(buf, j)
                dims.append(x)
        elif fn == 2 and wt == 0:
            dt = v
        elif fn == 8 and wt == 2:
            name = buf[span[0]:span[1]].decode("utf-8", "replace")
        elif fn == 9 and wt == 2:
            raw = (span[0], span[1])
        elif fn == 13 and wt == 2:
            ext = ext or []
            ext.append(span)
    return {"name": name, "dims": dims, "data_type": dt,
            "raw_span": raw, "has_external": bool(ext)}


def onnx_initializers_bytes(buf):
    out = []
    n = len(buf)
    for fn, wt, v, span in _pb_iter(buf, 0, n):
        if fn == 7 and wt == 2:  # graph
            gs, ge = span
            for gfn, gwt, gv, gspan in _pb_iter(buf, gs, ge):
                if gfn == 5 and gwt == 2:  # initializer
                    out.append(_tensorproto(buf, gspan[0], gspan[1]))
                elif gfn == 1 and gwt == 2:  # node -> subgraph attrs (rare)
                    pass
    return out


def onnx_summary(inits):
    learned = 0
    nonlearned = 0
    by_dt = {}
    for t in inits:
        p = prod(t["dims"]) if t["dims"] else 1
        dn = ONNX_DT.get(t["data_type"], ("DT%s" % t["data_type"], 0))[0]
        by_dt[dn] = by_dt.get(dn, 0) + p
        if dn in ("INT64", "INT32", "BOOL", "UINT8", "INT8", "UINT16", "INT16",
                  "UINT32", "UINT64"):
            nonlearned += p
        else:
            learned += p
    return {"learned": learned, "non_learned": nonlearned,
            "total": learned + nonlearned, "by_dtype": by_dt,
            "n_initializers": len(inits)}


def fmt(n):
    return "{:,}".format(n)
