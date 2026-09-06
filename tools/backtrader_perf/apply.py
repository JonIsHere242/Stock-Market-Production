"""Apply the backtrader perf edits (see README.md) by exact string replacement
to the installed package (or to the directory given as argv[1]).
Every `old` must occur exactly once, so re-running on a patched tree fails
loudly instead of double-patching."""
import io, os, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__import__('backtrader').__file__)


def patch(rel, pairs):
    path = os.path.join(ROOT, rel)
    s = io.open(path, encoding='utf-8', newline='').read()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, (rel, n, old[:80])
        s = s.replace(old, new)
    io.open(path, 'w', encoding='utf-8', newline='').write(s)
    print('patched', rel, len(pairs))


TRIPLE = "'" * 3

# ---------------- linebuffer.py ----------------
lb = [
("import array\nimport collections\nimport datetime\nfrom itertools import islice\nimport math\n",
 "import array\nimport collections\nimport datetime\nfrom itertools import islice, repeat\nimport math\n"),
("    def __getitem__(self, ago):\n        return self.array[self.idx + ago]\n",
 "    def __getitem__(self, ago):\n        return self.array[self._idx + ago]\n"),
("        if self.useislice:\n            start = self.idx + ago - size + 1\n            end = self.idx + ago + 1\n            return list(islice(self.array, start, end))\n\n        return self.array[self.idx + ago - size + 1:self.idx + ago + 1]\n",
 "        idx = self._idx\n        if self.useislice:\n            start = idx + ago - size + 1\n            end = idx + ago + 1\n            return list(islice(self.array, start, end))\n\n        return self.array[idx + ago - size + 1:idx + ago + 1]\n"),
("            value (variable): value to be set\n        " + TRIPLE + "\n        self.array[self.idx + ago] = value\n        for binding in self.bindings:\n            binding[ago] = value\n\n    def set(self, value, ago=0):",
 "            value (variable): value to be set\n        " + TRIPLE + "\n        self.array[self._idx + ago] = value\n        for binding in self.bindings:\n            binding[ago] = value\n\n    def set(self, value, ago=0):"),
("            the slice\n        " + TRIPLE + "\n        self.array[self.idx + ago] = value\n        for binding in self.bindings:\n            binding[ago] = value\n\n    def home(self):",
 "            the slice\n        " + TRIPLE + "\n        self.array[self._idx + ago] = value\n        for binding in self.bindings:\n            binding[ago] = value\n\n    def home(self):"),
("        self.idx += size\n        self.lencount += size\n\n        for i in range(size):\n            self.array.append(value)\n",
 "        if self.mode == self.UnBounded:\n            self._idx += size\n        else:\n            self.set_idx(self._idx + size)\n        self.lencount += size\n\n        if size == 1:\n            self.array.append(value)\n        else:\n            self.array.extend(repeat(value, size))\n"),
("    def rewind(self, size=1):\n        self.idx -= size\n        self.lencount -= size\n",
 "    def rewind(self, size=1):\n        if self.mode == self.UnBounded:\n            self._idx -= size\n        else:\n            self.set_idx(self._idx - size)\n        self.lencount -= size\n"),
("            size (int): How many extra positions to move forward\n        " + TRIPLE + "\n        self.idx += size\n        self.lencount += size\n",
 "            size (int): How many extra positions to move forward\n        " + TRIPLE + "\n        if self.mode == self.UnBounded:\n            self._idx += size\n        else:\n            self.set_idx(self._idx + size)\n        self.lencount += size\n"),
("        self.extension += size\n        for i in range(size):\n            self.array.append(value)\n",
 "        self.extension += size\n        if size == 1:\n            self.array.append(value)\n        elif size > 1:\n            self.array.extend(repeat(value, size))\n"),
("        return num2date(self.array[self.idx + ago],\n                        tz=tz or self._tz, naive=naive)\n\n    def date(",
 "        return num2date(self.array[self._idx + ago],\n                        tz=tz or self._tz, naive=naive)\n\n    def date("),
("        return num2date(self.array[self.idx + ago],\n                        tz=tz or self._tz, naive=naive).date()\n",
 "        return num2date(self.array[self._idx + ago],\n                        tz=tz or self._tz, naive=naive).date()\n"),
("        return num2date(self.array[self.idx + ago],\n                        tz=tz or self._tz, naive=naive).time()\n",
 "        return num2date(self.array[self._idx + ago],\n                        tz=tz or self._tz, naive=naive).time()\n"),
("        return math.trunc(self.array[self.idx + ago])\n", "        return math.trunc(self.array[self._idx + ago])\n"),
("        return math.modf(self.array[self.idx + ago])[0]\n", "        return math.modf(self.array[self._idx + ago])[0]\n"),
("        return time2num(num2date(self.array[self.idx + ago]).time())\n", "        return time2num(num2date(self.array[self._idx + ago]).time())\n"),
("        return int(self.array[self.idx + ago]) + tm\n", "        return int(self.array[self._idx + ago]) + tm\n"),
("        return num2date(int(self.array[self.idx + ago]) + tm)\n", "        return num2date(int(self.array[self._idx + ago]) + tm)\n"),
]
patch('linebuffer.py', lb)
p = os.path.join(ROOT, 'linebuffer.py')
s = io.open(p, encoding='utf-8', newline='').read()
old = "        dtime = self.array[self.idx + ago]\n"
assert s.count(old) == 5, s.count(old)
s = s.replace(old, "        dtime = self.array[self._idx + ago]\n")
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('patched linebuffer.py tm_* reads')

# ---------------- lineseries.py ----------------
patch('lineseries.py', [
("    def __len__(self):\n        " + TRIPLE + "\n        Proxy line operation\n        " + TRIPLE + "\n        return len(self.lines[0])\n",
 "    def __len__(self):\n        " + TRIPLE + "\n        Proxy line operation\n        " + TRIPLE + "\n        return self.lines[0].lencount\n"),
("    def __len__(self):\n        return len(self.lines)\n\n    def __getitem__(self, key):\n        return self.lines[0][key]\n",
 "    def __len__(self):\n        return self.lines.lines[0].lencount\n\n    def __getitem__(self, key):\n        return self.lines[0][key]\n"),
("        for l, line in enumerate(_obj.lines):\n            setattr(_obj, 'line_%s' % l, _obj._getlinealias(l))\n            setattr(_obj, 'line_%d' % l, line)\n            setattr(_obj, 'line%d' % l, line)\n\n        # Parameter values have now been set before __init__\n        return _obj, args, kwargs\n",
 "        for l, line in enumerate(_obj.lines):\n            setattr(_obj, 'line_%s' % l, _obj._getlinealias(l))\n            setattr(_obj, 'line_%d' % l, line)\n            setattr(_obj, 'line%d' % l, line)\n\n        # PERF: direct references to the named lines. Reading data.close\n        # used to go LineSeries.__getattr__ -> getattr(lines, name) ->\n        # LineAlias.__get__ on every access; an instance attribute is one\n        # dictionary hit. Only aliases the class does not already define are\n        # published, so attribute resolution order is unchanged for the rest.\n        # LineSeriesStub replaces self.lines in __init__ but declares no\n        # named lines, so nothing stale is published for it.\n        lines_obj = _obj.lines\n        for l, linealias in enumerate(lines_obj._getlines()):\n            if linealias and not hasattr(cls, linealias):\n                setattr(_obj, linealias, lines_obj.lines[l])\n\n        # Parameter values have now been set before __init__\n        return _obj, args, kwargs\n"),
])

# ---------------- feed.py ----------------
patch('feed.py', [
("        _obj._barstack = collections.deque()  # for filter operations\n        _obj._barstash = collections.deque()  # for filter operations\n",
 "        _obj._barstack = collections.deque()  # for filter operations\n        _obj._barstash = collections.deque()  # for filter operations\n\n        # PERF: the tick_xxx attribute names and their line indices, computed\n        # once instead of rebuilt through getlinealiases()/getattr on every\n        # bar of every data in _tick_nullify/_tick_fill\n        _aliases = _obj.getlinealiases()\n        _obj._tickattrs = tuple(('tick_' + a, i) for i, a in enumerate(_aliases)\n                                if a != 'datetime')\n        _obj._tickattr0 = 'tick_' + _obj._getlinealias(0)\n"),
("        for lalias in self.getlinealiases():\n            if lalias != 'datetime':\n                setattr(self, 'tick_' + lalias, None)\n\n        self.tick_last = None\n\n    def _tick_fill(self, force=False):\n        # If nothing filled the tick_xxx attributes, the bar is the tick\n        alias0 = self._getlinealias(0)\n        if force or getattr(self, 'tick_' + alias0, None) is None:\n            for lalias in self.getlinealiases():\n                if lalias != 'datetime':\n                    setattr(self, 'tick_' + lalias,\n                            getattr(self.lines, lalias)[0])\n\n            self.tick_last = getattr(self.lines, alias0)[0]\n\n    def advance_peek(self):\n        if len(self) < self.buflen():\n            return self.lines.datetime[1]  # return the future\n\n        return float('inf')  # max date else\n",
 "        for tname, _ in self._tickattrs:\n            setattr(self, tname, None)\n\n        self.tick_last = None\n\n    def _tick_fill(self, force=False):\n        # If nothing filled the tick_xxx attributes, the bar is the tick\n        if force or getattr(self, self._tickattr0, None) is None:\n            lines = self.lines.lines\n            for tname, i in self._tickattrs:\n                line = lines[i]\n                setattr(self, tname, line.array[line._idx])\n\n            line = lines[0]\n            self.tick_last = line.array[line._idx]\n\n    def advance_peek(self):\n        line0 = self.lines.lines[0]\n        if line0.lencount < len(line0.array) - line0.extension:\n            dtline = self.lines.datetime\n            return dtline.array[dtline._idx + 1]  # return the future\n\n        return float('inf')  # max date else\n"),
])

# ---------------- brokers/bbroker.py ----------------
patch('brokers/bbroker.py', [
("        for data in datas or self.positions:\n            comminfo = self.getcommissioninfo(data)\n            position = self.positions[data]\n",
 "        # PERF: when valuing the whole book, skip flat positions. Every\n        # comminfo term below is proportional to size, so a size-0 position\n        # contributes exactly 0.0 to each accumulator (adding 0.0 to a float\n        # is an identity), and the per-data early return only applies when\n        # datas was given.\n        skipflat = not datas\n        for data in datas or self.positions:\n            position = self.positions[data]\n            if skipflat and not position.size:\n                continue\n            comminfo = self.getcommissioninfo(data)\n"),
("        for data, pos in self.positions.items():\n            if pos:\n",
 "        for data, pos in self.positions.items():\n            if pos.size:\n"),
("        for data, pos in self.positions.items():\n            # futures change cash every bar\n            if pos:\n",
 "        for data, pos in self.positions.items():\n            # futures change cash every bar\n            if pos.size:\n"),
])

# ---------------- feeds/pandafeed.py ----------------
patch('feeds/pandafeed.py', [
("            self._colmapping[k] = v\n\n    def _load(self):\n        self._idx += 1\n\n        if self._idx >= len(self.p.dataname):\n            # exhausted all rows\n            return False\n\n        # Set the standard datafields\n        for datafield in self.getlinealiases():\n            if datafield == 'datetime':\n                continue\n\n            colindex = self._colmapping[datafield]\n            if colindex is None:\n                # datafield signaled as missing in the stream: skip it\n                continue\n\n            # get the line to be set\n            line = getattr(self.lines, datafield)\n\n            # indexing for pandas: 1st is colum, then row\n            line[0] = self.p.dataname.iloc[self._idx, colindex]\n\n        # datetime conversion\n        coldtime = self._colmapping['datetime']\n\n        if coldtime is None:\n            # standard index in the datetime\n            tstamp = self.p.dataname.index[self._idx]\n        else:\n            # it's in a different column ... use standard column index\n            tstamp = self.p.dataname.iloc[self._idx, coldtime]\n\n        # convert to float via datetime and store it\n        dt = tstamp.to_pydatetime()\n        dtnum = date2num(dt)\n        self.lines.datetime[0] = dtnum\n\n        # Done ... return\n        return True\n",
 "            self._colmapping[k] = v\n\n        # PERF: pull each mapped column out of the frame once. The original\n        # _load did DataFrame.iloc[row, col] per cell per bar, which is ~25\n        # microseconds of pandas indexing machinery for a single float and\n        # dominated the whole preload. Values are identical: iloc[:, col]\n        # keeps the column dtype, so element i is the same scalar\n        # iloc[i, col] would have produced.\n        df = self.p.dataname\n        self._nrows = len(df)\n        self._colarrays = []\n        for lidx, datafield in enumerate(self.getlinealiases()):\n            if datafield == 'datetime':\n                continue\n            colindex = self._colmapping[datafield]\n            if colindex is None:\n                continue  # datafield signaled as missing in the stream\n            self._colarrays.append((lidx, df.iloc[:, colindex].to_numpy()))\n\n        coldtime = self._colmapping['datetime']\n        if coldtime is None:\n            dtvalues = df.index\n        else:\n            dtvalues = df.iloc[:, coldtime]\n        # positional scalar access, same objects as index[i] /\n        # iloc[i, coldtime] (pandas Timestamps with to_pydatetime)\n        self._dtvalues = list(dtvalues)\n\n    def _load(self):\n        self._idx += 1\n        idx = self._idx\n\n        if idx >= self._nrows:\n            # exhausted all rows\n            return False\n\n        # Set the standard datafields\n        lines = self.lines.lines\n        for lidx, colarray in self._colarrays:\n            lines[lidx][0] = colarray[idx]\n\n        # datetime conversion: convert to float via datetime and store it\n        dt = self._dtvalues[idx].to_pydatetime()\n        dtnum = date2num(dt)\n        self.lines.datetime[0] = dtnum\n\n        # Done ... return\n        return True\n"),
])

# ---- second pass: feed.advance restructure, Lines.advance inline ----

patch('feed.py', [
("    def advance(self, size=1, datamaster=None, ticks=True):\n        if ticks:\n            self._tick_nullify()\n\n        # Need intercepting this call to support datas with\n        # different lengths (timeframes)\n        self.lines.advance(size)\n\n        if datamaster is not None:\n",
 "    def advance(self, size=1, datamaster=None, ticks=True):\n        if datamaster is None:\n            # PERF: the common preloaded/runonce path. Nullify-then-fill\n            # leaves the same state as a forced fill, and nothing reads the\n            # tick_xxx attributes in between, so only the branch that would\n            # have ended without a fill (past the last bar) still nullifies.\n            self.lines.advance(size)\n            if ticks:\n                if len(self) < self.buflen():\n                    self._tick_fill(force=True)\n                else:\n                    self._tick_nullify()\n            return\n\n        if ticks:\n            self._tick_nullify()\n\n        # Need intercepting this call to support datas with\n        # different lengths (timeframes)\n        self.lines.advance(size)\n\n        if datamaster is not None:\n"),
])
patch('lineseries.py', [
("    def advance(self, size=1):\n        '''\n        Proxy line operation\n        '''\n        for line in self.lines:\n            line.advance(size)\n",
 "    def advance(self, size=1):\n        '''\n        Proxy line operation\n        '''\n        # PERF: inline LineBuffer.advance for unbounded buffers (a method\n        # call per line per bar otherwise); QBuffer lines keep the call so\n        # set_idx's clamp semantics are untouched\n        for line in self.lines:\n            if line.mode == 0:  # LineBuffer.UnBounded\n                line._idx += size\n                line.lencount += size\n            else:\n                line.advance(size)\n"),
])

# ---- third pass: Python 3.10+ collections.abc crash fix ----
# Python 3.10 removed the ABC aliases from the top-level collections module;
# these four sites raised AttributeError (optstrategy with iterables, bindlines
# with an iterable, any csv writer at stop). Pure crash fix, no perf intent.
patch('cerebro.py', [("            elif not isinstance(elem, collections.Iterable):", "            elif not isinstance(elem, collections.abc.Iterable):")])
patch('lineiterator.py', [
 ("        elif not isinstance(owner, collections.Iterable):", "        elif not isinstance(owner, collections.abc.Iterable):"),
 ("        elif not isinstance(own, collections.Iterable):", "        elif not isinstance(own, collections.abc.Iterable):")])
patch('writer.py', [("            elif isinstance(val, (list, tuple, collections.Iterable)):", "            elif isinstance(val, (list, tuple, collections.abc.Iterable)):")])
print('all patches applied')
