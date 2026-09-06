"""
Drop-in replacement for ``from tqdm import tqdm`` that never redraws.

tqdm repaints its bar in place many times a second. Anything that captures the
stream instead of rendering it (the nightly step logs, PowerShell tee/transcript,
CI) turns every repaint into its own line, so one stage emits thousands of junk
lines. This shim keeps the tqdm call-site API but prints at most ~20 plain,
newline-terminated progress lines per bar: one every 5% when the total is known,
else one every 10 seconds, plus a final summary line on close.

Usage: replace ``from tqdm import tqdm`` with ``from auxiliary._quiet_progress import tqdm``.
No call-site changes needed. Layout kwargs (ncols, bar_format, position, leave,
mininterval, ...) are accepted and ignored.
"""

import sys
import time

_PCT_STEP    = 5      # print every N percent when total is known
_TIME_STEP_S = 10.0   # print every N seconds when total is unknown


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class tqdm:
    def __init__(self, iterable=None, desc=None, total=None, unit="it",
                 disable=False, file=None, **_ignored):
        self.iterable = iterable
        self.desc     = desc or ""
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        self.total   = total
        self.n       = 0
        self.postfix = ""
        self.unit    = unit
        self.disable = disable
        self._file   = file if file is not None else sys.stderr
        self._start  = time.time()
        self._last_t = self._start
        self._next_pct = _PCT_STEP
        self._closed = False

    # -- iterable wrapping:  for x in tqdm(items, ...) ----------------------
    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        self.close()

    # -- context manager:  with tqdm(...) as bar ----------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- core API -----------------------------------------------------------
    def update(self, n=1):
        self.n += n
        if self.disable:
            return
        now = time.time()
        if self.total:
            pct = self.n / self.total * 100.0
            if pct >= self._next_pct or self.n >= self.total:
                while self._next_pct <= pct:
                    self._next_pct += _PCT_STEP
                self._emit(now)
        elif now - self._last_t >= _TIME_STEP_S:
            self._emit(now)

    def set_description(self, desc=None, refresh=True):
        self.desc = desc or ""

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        items = list((ordered_dict or {}).items()) + list(kwargs.items())
        self.postfix = ", ".join(f"{k}={v}" for k, v in items)

    def close(self):
        if self._closed or self.disable:
            return
        self._closed = True
        elapsed = time.time() - self._start
        total   = f"/{self.total}" if self.total else f" {self.unit}"
        print(f"  {self.desc or 'progress'}: {self.n}{total} done in "
              f"{_fmt_elapsed(elapsed)}", file=self._file, flush=True)

    def refresh(self):
        pass

    @classmethod
    def write(cls, s, file=None, end="\n", nolock=False):
        print(s, file=file if file is not None else sys.stdout, end=end, flush=True)

    # -- internal -----------------------------------------------------------
    def _emit(self, now):
        self._last_t = now
        elapsed = _fmt_elapsed(now - self._start)
        if self.total:
            pct  = self.n / self.total * 100.0
            head = f"{self.n}/{self.total} ({pct:3.0f}%)"
        else:
            head = f"{self.n} {self.unit}"
        tail = f"  {self.postfix}" if self.postfix else ""
        desc = f"{self.desc}: " if self.desc else ""
        print(f"  {desc}{head}  [{elapsed}]{tail}", file=self._file, flush=True)


def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)
