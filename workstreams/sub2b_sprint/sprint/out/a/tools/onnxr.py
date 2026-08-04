"""Remote ONNX initializer enumeration over HTTP Range.

An ONNX file is a ModelProto. Weights live in GraphProto.initializer (field 5)
as TensorProto, whose fields serialise in field-number order:
    1 dims (int64, packed or repeated)   2 data_type (int32)
    8 name (string)                      9 raw_data (bytes)  <- the payload
   13 external_data                     14 data_location
Because `raw_data` is the LAST big field, we can read the small leading fields
and then SEEK PAST the payload with a fresh Range request. Total bytes pulled is
kilobytes, not gigabytes -- no weights are downloaded.

Also enumerates Constant-node attribute tensors (field 1 = node, attribute 5 =
AttributeProto.t), which is where some exporters hide weights.
"""
import json
import sys

import net

ONNX_DT = {1: ("FLOAT", 4), 2: ("UINT8", 1), 3: ("INT8", 1), 4: ("UINT16", 2),
           5: ("INT16", 2), 6: ("INT32", 4), 7: ("INT64", 8), 9: ("BOOL", 1),
           10: ("FLOAT16", 2), 11: ("DOUBLE", 8), 12: ("UINT32", 4),
           13: ("UINT64", 8), 14: ("COMPLEX64", 8), 15: ("COMPLEX128", 16),
           16: ("BFLOAT16", 2), 17: ("FLOAT8E4M3FN", 1), 18: ("FLOAT8E4M3FNUZ", 1),
           19: ("FLOAT8E5M2", 1), 20: ("FLOAT8E5M2FNUZ", 1), 21: ("UINT4", 0.5),
           22: ("INT4", 0.5), 23: ("FLOAT4E2M1", 0.5)}

INT_DT = {"UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "BOOL",
          "UINT32", "UINT64"}


class Reader:
    """Sequential reader over a remote or local file with seek-forward."""

    def __init__(self, url=None, path=None, size=None, chunk=1 << 18):
        self.url, self.path, self.chunk = url, path, chunk
        self.buf = b""
        self.bufpos = 0        # absolute file offset of buf[0]
        self.pos = 0           # absolute cursor
        self.fetched = 0
        if path:
            self.fh = open(path, "rb")
            import os
            self.size = os.path.getsize(path)
        else:
            self.fh = None
            self.size = size if size is not None else self._probe_size()

    def _probe_size(self):
        r = net.S.get(self.url, headers={"Range": "bytes=0-0"},
                      allow_redirects=True, timeout=120)
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            return int(cr.split("/")[-1])
        return int(r.headers.get("Content-Length", 0))

    def _fill(self, n):
        while self.pos + n > self.bufpos + len(self.buf):
            start = self.bufpos + len(self.buf)
            want = max(self.chunk, n)
            end = min(start + want, self.size) - 1
            if end < start:
                raise EOFError("eof at %d" % start)
            if self.fh:
                self.fh.seek(start)
                blk = self.fh.read(end - start + 1)
            else:
                blk = net.rng(self.url, start, end)
            self.fetched += len(blk)
            self.buf += blk
            if len(self.buf) > (8 << 20):   # trim consumed prefix
                drop = self.pos - self.bufpos
                self.buf = self.buf[drop:]
                self.bufpos = self.pos

    def peek(self, n):
        self._fill(n)
        o = self.pos - self.bufpos
        return self.buf[o:o + n]

    def read(self, n):
        b = self.peek(n)
        self.pos += n
        return b

    def skip(self, n):
        """Seek forward; drops the buffer so no payload bytes are fetched."""
        self.pos += n
        if self.pos > self.bufpos + len(self.buf):
            self.buf = b""
            self.bufpos = self.pos

    def byte(self):
        b = self.peek(1)
        self.pos += 1
        return b[0]

    def varint(self):
        r = 0
        s = 0
        while True:
            b = self.byte()
            r |= (b & 0x7F) << s
            if not (b & 0x80):
                return r
            s += 7

    def tag(self):
        k = self.varint()
        return k >> 3, k & 7


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _skip_field(rd, wt):
    if wt == 0:
        rd.varint()
    elif wt == 1:
        rd.skip(8)
    elif wt == 2:
        rd.skip(rd.varint())
    elif wt == 5:
        rd.skip(4)
    else:
        raise ValueError("wiretype %d" % wt)


def read_tensorproto(rd, end):
    dims, dt, name, raw_len, ext = [], None, "", 0, None
    while rd.pos < end:
        fn, wt = rd.tag()
        if fn == 1 and wt == 0:
            dims.append(rd.varint())
        elif fn == 1 and wt == 2:              # packed dims
            n = rd.varint()
            stop = rd.pos + n
            while rd.pos < stop:
                dims.append(rd.varint())
        elif fn == 2 and wt == 0:
            dt = rd.varint()
        elif fn == 8 and wt == 2:
            n = rd.varint()
            name = rd.read(n).decode("utf-8", "replace")
        elif fn == 9 and wt == 2:              # raw_data -- SKIP the payload
            raw_len = rd.varint()
            rd.skip(raw_len)
        elif fn == 13 and wt == 2:             # external_data
            n = rd.varint()
            blob = rd.read(n)
            ext = (ext or []) + [blob.decode("utf-8", "replace")]
        elif fn in (4, 5, 6, 7, 10) and wt == 2:   # packed float/int32/... data
            n = rd.varint()
            raw_len += n
            rd.skip(n)
        else:
            _skip_field(rd, wt)
    rd.pos = end
    return {"name": name, "dims": dims, "data_type": dt,
            "raw_bytes": raw_len, "external": ext}


def read_attribute(rd, end, out_tensors):
    """AttributeProto: 1 name, 20 type, 5 t (TensorProto), 6 g (GraphProto)."""
    aname = ""
    while rd.pos < end:
        fn, wt = rd.tag()
        if fn == 1 and wt == 2:
            n = rd.varint()
            aname = rd.read(n).decode("utf-8", "replace")
        elif fn == 5 and wt == 2:              # single tensor
            n = rd.varint()
            t = read_tensorproto(rd, rd.pos + n)
            t["source"] = "constant_attr:" + aname
            out_tensors.append(t)
        elif fn == 6 and wt == 2:              # subgraph
            n = rd.varint()
            read_graph(rd, rd.pos + n, out_tensors, sub=True)
        elif fn == 10 and wt == 2:             # repeated tensors
            n = rd.varint()
            t = read_tensorproto(rd, rd.pos + n)
            t["source"] = "attr_tensors:" + aname
            out_tensors.append(t)
        else:
            _skip_field(rd, wt)
    rd.pos = end


def read_node(rd, end, out_tensors):
    while rd.pos < end:
        fn, wt = rd.tag()
        if fn == 5 and wt == 2:                # attribute
            n = rd.varint()
            read_attribute(rd, rd.pos + n, out_tensors)
        else:
            _skip_field(rd, wt)
    rd.pos = end


def read_graph(rd, end, out_tensors, sub=False):
    while rd.pos < end:
        fn, wt = rd.tag()
        if fn == 1 and wt == 2:                # node
            n = rd.varint()
            read_node(rd, rd.pos + n, out_tensors)
        elif fn == 5 and wt == 2:              # initializer
            n = rd.varint()
            t = read_tensorproto(rd, rd.pos + n)
            t["source"] = "subgraph_initializer" if sub else "initializer"
            out_tensors.append(t)
        elif fn == 15 and wt == 2:             # sparse_initializer
            n = rd.varint()
            t = read_tensorproto(rd, rd.pos + n)
            t["source"] = "sparse_initializer"
            out_tensors.append(t)
        else:
            _skip_field(rd, wt)
    rd.pos = end


def enumerate_onnx(url=None, path=None):
    rd = Reader(url=url, path=path)
    tensors = []
    while rd.pos < rd.size:
        try:
            fn, wt = rd.tag()
        except EOFError:
            break
        if fn == 7 and wt == 2:                # graph
            n = rd.varint()
            read_graph(rd, rd.pos + n, tensors)
        else:
            _skip_field(rd, wt)
    return tensors, rd


def summarise(tensors, dedup_by_name=True):
    seen = set()
    learned = nonlearned = 0
    by_dt = {}
    raw = 0
    kept = []
    for t in tensors:
        key = (t["name"], tuple(t["dims"]), t["data_type"])
        if dedup_by_name and t["name"] and key in seen:
            continue
        seen.add(key)
        kept.append(t)
        p = prod(t["dims"])
        dn = ONNX_DT.get(t["data_type"], ("DT%s" % t["data_type"], 0))[0]
        by_dt[dn] = by_dt.get(dn, 0) + p
        raw += t["raw_bytes"]
        if dn in INT_DT:
            nonlearned += p
        else:
            learned += p
    return {"n_tensors": len(kept), "n_raw_entries": len(tensors),
            "learned": learned, "non_learned": nonlearned,
            "total": learned + nonlearned, "by_dtype": by_dt,
            "raw_data_bytes": raw}


def report(name, url=None, path=None):
    ts, rd = enumerate_onnx(url=url, path=path)
    s = summarise(ts)
    s["file_bytes"] = rd.size
    s["bytes_fetched"] = rd.fetched
    s["name"] = name
    s["coverage"] = round(s["raw_data_bytes"] / rd.size, 6) if rd.size else 0
    return s, ts


if __name__ == "__main__":
    s, ts = report(sys.argv[1], url=sys.argv[2] if sys.argv[2].startswith("http")
                   else None,
                   path=None if sys.argv[2].startswith("http") else sys.argv[2])
    print(json.dumps(s, indent=1))
