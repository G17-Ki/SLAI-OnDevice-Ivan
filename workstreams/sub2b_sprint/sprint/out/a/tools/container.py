"""Remote container walks: ZIP (torch.save v1.5+ / .pt / .bin) and TAR (.nemo).

The point: earn EXACT without downloading weights.

A torch.save file is a ZIP whose central directory lists one member per tensor
storage plus a small `data.pkl`. We
  1. Range-read the tail -> End Of Central Directory -> central directory offset
  2. Range-read the central directory -> member names + sizes
  3. Range-read ONLY `data.pkl` (a few hundred KB)
  4. Unpickle it with stubbed rebuild functions, so every
     `_rebuild_tensor_v2(storage, offset, size, stride, ...)` yields the exact
     shape, and `persistent_load` yields the exact dtype.
That is a tensor-level parse == EXACT, at a cost of well under 1 MB.

A `.nemo` is an (unpacked) TAR; we walk 512-byte headers to list members, then
recurse into the contained `model_weights.ckpt` with the ZIP walk.
"""
import io
import json
import pickle
import struct
import sys
import zlib

import net

TORCH_DT = {
    "FloatStorage": ("float32", 4), "DoubleStorage": ("float64", 8),
    "HalfStorage": ("float16", 2), "BFloat16Storage": ("bfloat16", 2),
    "LongStorage": ("int64", 8), "IntStorage": ("int32", 4),
    "ShortStorage": ("int16", 2), "CharStorage": ("int8", 1),
    "ByteStorage": ("uint8", 1), "BoolStorage": ("bool", 1),
    "ComplexFloatStorage": ("complex64", 8),
    "ComplexDoubleStorage": ("complex128", 16),
    "QInt8Storage": ("qint8", 1), "QUInt8Storage": ("quint8", 1),
    "Float8_e4m3fnStorage": ("float8_e4m3fn", 1),
    "Float8_e5m2Storage": ("float8_e5m2", 1),
}
INT_DT = {"int64", "int32", "int16", "int8", "uint8", "bool"}


# ------------------------------------------------------------------ ZIP

def zip_central_directory(url, size=None):
    if isinstance(url, SubFile):
        size = url.length
    if size is None:
        r = net.S.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True,
                      timeout=120)
        size = int(r.headers["Content-Range"].split("/")[-1])
    tail = _rng(url, max(0, size - 70000), size - 1)
    # EOCD64 first (torch files are routinely >4 GB)
    i = tail.rfind(b"PK\x06\x06")
    if i >= 0:
        cd_size, cd_off = struct.unpack("<QQ", tail[i + 40:i + 56])
    else:
        j = tail.rfind(b"PK\x05\x06")
        if j < 0:
            raise ValueError("no EOCD")
        cd_size, cd_off = struct.unpack("<II", tail[j + 12:j + 20])
    cd = _rng(url, cd_off, cd_off + cd_size - 1)
    members = []
    p = 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        f = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        method, csize, usize = f[4], f[8], f[9]
        nlen, elen, clen, lho = f[10], f[11], f[12], f[16]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or lho == 0xFFFFFFFF:
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    vals = []
                    k = q + 4
                    while k + 8 <= q + 4 + hsz:
                        vals.append(struct.unpack("<Q", extra[k:k + 8])[0])
                        k += 8
                    it = iter(vals)
                    if usize == 0xFFFFFFFF:
                        usize = next(it)
                    if csize == 0xFFFFFFFF:
                        csize = next(it)
                    if lho == 0xFFFFFFFF:
                        lho = next(it)
                q += 4 + hsz
        members.append({"name": name, "method": method, "csize": csize,
                        "usize": usize, "local_header_offset": lho})
        p += 46 + nlen + elen + clen
    return members, size


def zip_read_member(url, m):
    lh = _rng(url, m["local_header_offset"], m["local_header_offset"] + 29)
    nlen, elen = struct.unpack("<HH", lh[26:30])
    start = m["local_header_offset"] + 30 + nlen + elen
    raw = _rng(url, start, start + m["csize"] - 1)
    if m["method"] == 0:
        return raw
    if m["method"] == 8:
        return zlib.decompress(raw, -15)
    raise ValueError("zip method %d" % m["method"])


class Node:
    """Stand-in for any reconstructed object. Supports the pickle BUILD opcode
    (__setstate__ / __dict__ update) and __reduce__ append/extend."""

    def __init__(self, call=None, args=(), kwargs=None):
        object.__setattr__(self, "_call", call)
        object.__setattr__(self, "_args", tuple(args))
        object.__setattr__(self, "_kwargs", dict(kwargs or {}))
        object.__setattr__(self, "_state", None)
        object.__setattr__(self, "_items", [])

    def __setstate__(self, state):
        object.__setattr__(self, "_state", state)

    def append(self, x):
        self._items.append(x)

    def extend(self, xs):
        self._items.extend(xs)

    def __setitem__(self, k, v):
        self._items.append((k, v))

    def __repr__(self):
        return "<Node %s>" % (self._call,)


_CLS = {}


def _make_class(mod, name):
    """A real type (pickle NEWOBJ requires one) that constructs a Node."""
    key = (mod, name)
    if key in _CLS:
        return _CLS[key]

    class Reconstructed(Node):
        _mod = mod
        _name = name

        def __new__(cls, *a, **k):
            o = object.__new__(cls)
            Node.__init__(o, name, a, k)
            return o

        def __init__(self, *a, **k):
            pass

    Reconstructed.__name__ = name
    Reconstructed.__qualname__ = name
    _CLS[key] = Reconstructed
    return Reconstructed


class _Stub:
    def __init__(self, mod, name):
        self.mod, self.name = mod, name

    def __call__(self, *a, **k):
        return Node(self.name, a, k)

    def __repr__(self):
        return "<%s.%s>" % (self.mod, self.name)


class MetaUnpickler(pickle.Unpickler):
    """Unpickles a torch state_dict without torch, without loading storages."""

    def __init__(self, f):
        super().__init__(f, encoding="latin1")
        self.storages = {}

    def find_class(self, mod, name):
        return _make_class(mod, name)

    def persistent_load(self, pid):
        # ('storage', <StorageCls>, key, location, numel)
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            cls = pid[1]
            cname = getattr(cls, "_name", None) or getattr(cls, "name", None) \
                or getattr(cls, "__name__", str(cls))
            rec = {"__storage__": True, "dtype_cls": cname, "key": str(pid[2]),
                   "location": pid[3], "numel": pid[4]}
            self.storages[str(pid[2])] = rec
            return rec
        return {"__persid__": repr(pid)}


def _walk(obj, prefix, out, depth=0, seen=None):
    if depth > 40:
        return
    if seen is None:
        seen = set()
    if isinstance(obj, Node):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        call = obj._call or ""
        if call.startswith("_rebuild_tensor") or call.startswith("_rebuild_parameter"):
            a = obj._args
            if call.startswith("_rebuild_parameter"):
                _walk(a[0] if a else None, prefix, out, depth + 1, seen)
                return
            storage = a[0] if a else None
            shape = a[2] if len(a) > 2 else []
            dtype = None
            if isinstance(storage, dict) and storage.get("__storage__"):
                dtype = TORCH_DT.get(storage["dtype_cls"],
                                     (storage["dtype_cls"], 0))[0]
            out.append({"name": prefix,
                        "shape": list(shape) if shape else [],
                        "dtype": dtype,
                        "storage_key": storage.get("key") if isinstance(storage, dict) else None,
                        "storage_numel": storage.get("numel") if isinstance(storage, dict) else None})
            return
        for i, x in enumerate(obj._args):
            _walk(x, "%s|a%d" % (prefix, i), out, depth + 1, seen)
        if obj._state is not None:
            _walk(obj._state, prefix, out, depth + 1, seen)
        for it in obj._items:
            if isinstance(it, tuple) and len(it) == 2:
                _walk(it[1], "%s.%s" % (prefix, it[0]) if prefix else str(it[0]),
                      out, depth + 1, seen)
            else:
                _walk(it, prefix, out, depth + 1, seen)
        return
    if isinstance(obj, dict):
        if obj.get("__storage__"):
            out.append({"name": prefix, "shape": None,
                        "dtype": TORCH_DT.get(obj["dtype_cls"],
                                              (obj["dtype_cls"], 0))[0],
                        "storage_key": obj["key"],
                        "storage_numel": obj["numel"]})
            return
        for k, v in obj.items():
            _walk(v, "%s.%s" % (prefix, k) if prefix else str(k), out,
                  depth + 1, seen)
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            _walk(x, "%s[%d]" % (prefix, i), out, depth + 1, seen)


def torch_zip_tensors(url, size=None, pkl_hint="data.pkl"):
    members, size = zip_central_directory(url, size)
    pkls = [m for m in members if m["name"].endswith(pkl_hint)]
    if not pkls:
        raise ValueError("no %s in zip; members: %s" %
                         (pkl_hint, [m["name"] for m in members[:8]]))
    pkls.sort(key=lambda m: len(m["name"]))
    blob = zip_read_member(url, pkls[0])
    up = MetaUnpickler(io.BytesIO(blob))
    obj = up.load()
    tensors = []
    _walk(obj, "", tensors)
    return {"members": members, "file_bytes": size, "pkl": pkls[0]["name"],
            "pkl_bytes": len(blob), "tensors": tensors,
            "storages": up.storages, "root": obj}


def prod(s):
    p = 1
    for x in (s or []):
        p *= x
    return p


def summarise_torch(res, dedup_storage=True):
    """Count params. Dedup by storage key so a tied/shared storage counts once."""
    seen = set()
    learned = nonlearned = 0
    by_dt = {}
    rows = []
    shared = []
    for t in res["tensors"]:
        if t["shape"] is None:
            continue
        key = t["storage_key"]
        if dedup_storage and key is not None and key in seen:
            shared.append(t)
            continue
        if key is not None:
            seen.add(key)
        p = prod(t["shape"])
        dt = t["dtype"] or "?"
        by_dt[dt] = by_dt.get(dt, 0) + p
        if dt in INT_DT:
            nonlearned += p
        else:
            learned += p
        rows.append(t)
    # storage-byte reconciliation from the zip central directory
    stor_bytes = sum(m["usize"] for m in res["members"]
                     if "/data/" in m["name"] or m["name"].endswith(".storage"))
    return {"n_tensors": len(rows), "n_shared_skipped": len(shared),
            "learned": learned, "non_learned": nonlearned,
            "total": learned + nonlearned, "by_dtype": by_dt,
            "file_bytes": res["file_bytes"], "pkl_bytes": res["pkl_bytes"],
            "storage_member_bytes": stor_bytes,
            "shared_storage_tensors": [s["name"] for s in shared][:40]}


class SubFile:
    """A byte window inside a remote file -- lets the ZIP walker run against a
    member embedded in a TAR (i.e. model_weights.ckpt inside a .nemo)."""

    def __init__(self, url, base, length):
        self.url, self.base, self.length = url, base, length


def _rng(u, a, b):
    if isinstance(u, SubFile):
        return net.rng(u.url, u.base + a, u.base + b)
    return net.rng(u, a, b)


# ------------------------------------------------------------------ TAR

def tar_members(url, size=None, max_scan=64):
    if size is None:
        r = net.S.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True,
                      timeout=120)
        size = int(r.headers["Content-Range"].split("/")[-1])
    out = []
    off = 0
    for _ in range(max_scan):
        if off + 512 > size:
            break
        h = _rng(url, off, off + 511)
        if h[:512] == b"\0" * 512:
            break
        name = h[0:100].split(b"\0")[0].decode("utf-8", "replace")
        if not name:
            break
        try:
            sz = int(h[124:136].split(b"\0")[0].strip() or b"0", 8)
        except ValueError:
            break
        typ = chr(h[156]) if h[156] else "0"
        prefix = h[345:500].split(b"\0")[0].decode("utf-8", "replace")
        full = prefix + "/" + name if prefix else name
        out.append({"name": full, "size": sz, "type": typ, "data_offset": off + 512})
        off += 512 + ((sz + 511) // 512) * 512
    return out, size
