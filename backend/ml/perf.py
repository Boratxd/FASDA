
import time
from datetime import datetime

class PerfTimer:

    WIDTH = 64

    def __init__(self, title: str = "Performance"):
        self.title       = title
        self._t0         = time.perf_counter()
        self._stage_t    = None
        self._stage_name = None
        self._stage_meta = ""
        self.stages: list[tuple[str, float, str]] = []

    def start(self, name: str, *, meta: str = "") -> None:

        self._stage_name = name
        self._stage_meta = meta
        self._stage_t    = time.perf_counter()

    def stop(self, *, meta: str | None = None) -> float:

        if self._stage_t is None:
            return 0.0
        elapsed = time.perf_counter() - self._stage_t
        m = meta if meta is not None else self._stage_meta
        self.stages.append((self._stage_name, elapsed, m))
        self._stage_t    = None
        self._stage_name = None
        self._stage_meta = ""
        return elapsed

    def lap(self, name: str, *, meta: str = "") -> float:

        elapsed = self.stop()
        self.start(name, meta=meta)
        return elapsed

    def total_elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def print_report(self, **summary) -> None:

        W     = self.WIDTH
        inner = W - 2
        total = self.total_elapsed()
        now   = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

        def top(l="╔", r="╗"):    return f"{l}{'═' * inner}{r}"
        def mid(l="╠", r="╣"):    return f"{l}{'═' * inner}{r}"
        def bot():                 return f"╚{'═' * inner}╝"
        def row(left, right=""):
            left_w  = inner - 16
            right_w = 15
            if right:
                s = f" {left:<{left_w}}{right:>{right_w}} "
            else:
                s = f" {left:<{inner - 2}} "

            s = s[:inner]
            return f"║{s:^{inner}}║" if not right else f"║{s}║"
        def row_plain(text):
            return f"║ {text:<{inner - 2}} ║"

        lines: list[str] = [
            top(),
            row_plain(f"FASDA · {self.title}"),
            row_plain(f"Started: {now}"),
            mid(),
        ]

        for name, elapsed, meta in self.stages:
            meta_str = f"  ({meta})" if meta else ""
            name_col = f"  {name}{meta_str}"
            time_col = f"{elapsed:>8.3f} s"

            avail = inner - 1 - len(time_col) - 1
            name_col = name_col[:avail].ljust(avail)
            lines.append(f"║ {name_col} {time_col} ║")

        lines.append(mid())

        total_col = f"{total:>8.3f} s"
        total_lbl = f"  {'TOTAL TIME':<{inner - 1 - len(total_col) - 1}}"
        lines.append(f"║{total_lbl} {total_col} ║")

        if summary:
            lines.append(mid())
            for k, v in summary.items():
                label = "  " + k.replace("_", " ").title()
                value = str(v)
                avail = inner - 1 - len(value) - 1
                label = label[:avail].ljust(avail)
                lines.append(f"║ {label} {value:>{len(value)}} ║")

        lines.append(bot())

        print("\n" + "\n".join(lines) + "\n", flush=True)
