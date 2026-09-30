import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from collections import Counter
from objects import Card
from colorama import Fore, Back, Style, init
from nn.timing import ModelTimer
from ddsolver.ddsrecorder import DDSRecorder

init()


def _load_dds3():
    """Import the dds3 extension (DDS 3.0.0 Python interface).

    Tries a normal import first (dds3 installed as a wheel). If that fails,
    falls back to a copy vendored under BEN's bin/ directory.
    """
    # Keep the real failure around. A vendored _dds3.so that exists but won't
    # load (e.g. built for a different Python version, or missing a dependent
    # shared library) raises ImportError here; swallowing it hides the cause.
    last_error = None
    try:
        import dds3
        return dds3
    except ImportError as ex:
        last_error = ex

    here = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == "win32":
        plat = "win"
    elif sys.platform == "darwin":
        plat = "darwin"
    else:
        plat = "linux"

    # Look for a vendored dds3 under bin/, covering the layouts BEN runs in:
    #   repo:              src/ddsolver/ddsolver.py -> <repo>/bin       (here/../..)
    #   flattened (Docker): /app/ddsolver/ddsolver.py -> /app/bin       (here/..)
    #   plus BEN_HOME/bin and <cwd>/bin as fallbacks.
    bin_roots = [
        os.path.join(os.path.abspath(os.path.join(here, "..", "..")), "bin"),
        os.path.join(os.path.abspath(os.path.join(here, "..")), "bin"),
    ]
    if os.getenv("BEN_HOME"):
        bin_roots.append(os.path.join(os.getenv("BEN_HOME"), "bin"))
    bin_roots.append(os.path.join(os.getcwd(), "bin"))

    seen = set()
    for root in bin_roots:
        for cand in (os.path.join(root, "dds3-" + plat),
                     os.path.join(root, "dds3"),
                     root):
            if cand in seen:
                continue
            seen.add(cand)
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
            try:
                import dds3
                return dds3
            except ImportError as ex:
                last_error = ex
                continue

    raise ImportError(
        "Could not import the 'dds3' extension (DDS 3.0.0 Python interface).\n"
        "Build it from the DDS repository and install or vendor it:\n"
        "  bazel build -c opt //python:dds3_wheel_dist   # produces a wheel in dist/\n"
        "  pip install dist/dds3-*.whl\n"
        f"or place the built dds3 package in BEN's bin/dds3-{plat}/ directory.\n"
        f"A _dds3.so built for a different Python than {sys.version_info.major}."
        f"{sys.version_info.minor} will fail to load here.\n"
        "See docs/python_interface.md in the DDS repository.\n"
        f"\nUnderlying import error: {last_error!r}"
    ) from last_error


dds3 = _load_dds3()


# DDS 3.0.0 removed internal multi-threading from the legacy batch API
# (SolveAllBoards now solves sequentially). The modern model is one
# SolverContext per worker thread; DDSolver parallelises with a thread pool,
# and solve_board_pbn releases the GIL during the solve, so the threads run
# concurrently. Each pool thread keeps its own SolverContext.
#
# Which "thread" that is depends on whether threading has been monkey-patched.
def _resolve_solver_parallelism():
    """Pick (executor class, thread-local class) for the DDS worker pool.

    gameapi.py calls monkey.patch_all() before importing this module, and that
    replaces threading.Thread with a greenlet. concurrent.futures reads
    threading.Thread at *call* time, so a ThreadPoolExecutor built afterwards
    starts every "worker" on the single OS thread the gevent hub occupies:
    max_workers=16, but max_simultaneous=1. solve_board_pbn releases the GIL and
    never yields to the hub, so a 200-board batch grinds through 200 solves back
    to back on one core (measured live: os_threads=1, cores_busy=0.98, 33.4 s).

    gevent.threadpool.ThreadPoolExecutor is documented as "a version of
    concurrent.futures.ThreadPoolExecutor that always uses native threads, even
    when threading is monkey-patched". Same submit/map/result and exception
    surface, so the call sites do not change; it wakes the waiting greenlet
    through the hub's async watcher, so the HTTP server keeps serving while a
    batch runs. Because those workers really are OS threads, the per-worker
    SolverContext has to be keyed by OS thread too — threading.local is gevent's
    greenlet-backed local once patched, and get_original returns the real
    thread._local.

    An unpatched process (the GUI, the SuitC worker, ddsreplay) already gets
    genuine OS threads from the stdlib executor, so it keeps both classes it has
    always used. The two must be decided together: greenlet workers need the
    greenlet-keyed local, OS-thread workers need the native one.
    """
    patched = False
    try:
        from gevent.monkey import get_original, is_module_patched
        patched = bool(is_module_patched("threading"))
    except Exception:
        patched = False

    if not patched:
        return ThreadPoolExecutor, threading.local

    try:
        from gevent.threadpool import ThreadPoolExecutor as _NativeThreadsExecutor
    except Exception:
        # A parallelism fix must never be able to stop the API from starting.
        return ThreadPoolExecutor, threading.local

    try:
        native_local = get_original("threading", "local")
    except Exception:
        native_local = threading.local
    return _NativeThreadsExecutor, native_local


_SolverPool, _SolverLocal = _resolve_solver_parallelism()
_ctx_local = _SolverLocal()


def _thread_context():
    ctx = getattr(_ctx_local, "ctx", None)
    if ctx is None:
        ctx = dds3.SolverContext()
        _ctx_local.ctx = ctx
    return ctx


# --- Temporary concurrency diagnostic ---------------------------------------
# Gated entirely by BEN_DDS_CONCURRENCY_PROBE, the same env-var convention as
# BEN_DDS_RECORD. With it unset, solve_helper takes its original code path and
# nothing below runs. It answers one question about a live batch: do the
# solve_board_pbn calls actually overlap in wall-clock time, or do they run
# back to back? Nothing here can change which card is chosen.
def _dds_probe_enabled():
    return os.environ.get("BEN_DDS_CONCURRENCY_PROBE", "").strip().lower() in ("1", "true", "yes", "on")


# threading.get_ident() is monkey-patched by gevent to a per-greenlet number, so
# it cannot distinguish an OS thread from a greenlet. This reads the kernel
# thread instead, and records in _OS_TID_KIND which source answered, so a
# fallback is never mistaken for a real OS identity.
_OS_TID_KIND = "unprobed"


def _os_thread_id():
    global _OS_TID_KIND
    try:
        import ctypes
        if sys.platform == "win32":
            _OS_TID_KIND = "GetCurrentThreadId"
            return int(ctypes.windll.kernel32.GetCurrentThreadId())
        libc = ctypes.CDLL(None)
        # gettid() needs glibc >= 2.30 (the image is Ubuntu 24.04). macOS has no
        # gettid, so the except branch flags the fallback instead of lying.
        libc.gettid.restype = ctypes.c_int
        _OS_TID_KIND = "gettid"
        return int(libc.gettid())
    except Exception:
        _OS_TID_KIND = "fallback-python-ident"
        return threading.get_ident()


# CPU seconds charged to the *calling OS thread*. Blocked time does not accrue,
# so this is what separates "several threads were inside a solve at the same
# time" from "several threads were waiting at the same time".
_CPU_CLOCK = getattr(time, "thread_time", None) or time.process_time
_CPU_CLOCK_NAME = "thread" if getattr(time, "thread_time", None) else "process"


def _dds_concurrency_report(dds, events):
    """Print one stderr line describing real concurrency for one solve batch.

    Reports counts and timings only — never PBN hands, cards, or any input data.

    Two independent signals, because neither one alone is a proof:
      os_threads / max_simultaneous — are there several kernel threads whose
        solve intervals overlap in wall-clock time?
      cores_busy = cpu / wall — how many cores were actually *running* code,
        from per-thread CPU time, which does not accrue while a thread blocks.
    spans is the sum of the per-board interval spans. It is reported for shape
    only and is deliberately NOT turned into a speedup number: if anything
    blocks inside the call (a lock, a native wait), every waiting thread's span
    covers that wait, spans pile up, and spans/wall reads as parallelism that is
    not happening. Verified locally: 16 threads serialised by a lock give
    max_simultaneous=16 and spans/wall ~15 while cores_busy ~0.9.
    """
    tids = {e[0] for e in events}
    idents = {e[1] for e in events}
    contexts = {e[2] for e in events}

    # Sweep line over interval boundaries. Ties sort end-before-start because
    # -1 < 1, so two back-to-back solves are not counted as overlapping.
    bounds = []
    for e in events:
        bounds.append((e[3], 1))
        bounds.append((e[4], -1))
    bounds.sort()
    active = 0
    peak = 0
    for _t, delta in bounds:
        active += delta
        if active > peak:
            peak = active

    wall = max(e[4] for e in events) - min(e[3] for e in events)
    work = sum(e[4] - e[3] for e in events)
    cpu = sum(e[6] - e[5] for e in events)

    try:
        from gevent.monkey import is_module_patched
        patched = str(bool(is_module_patched("threading")))
    except Exception:
        patched = "no-gevent"

    pool = getattr(dds, "_pool", None)
    live = getattr(pool, "_threads", None)
    sys.stderr.write(
        "DDS-CONCURRENCY boards=%d configured_max_threads=%s executor_max_workers=%s "
        "executor_workers_live=%s py_idents=%d os_tid_kind=%s os_threads=%d contexts=%d "
        "max_simultaneous=%d wall=%.3fs cpu=%.3fs cpu_clock=%s cores_busy=%.2f "
        "spans=%.3fs verdict=%s "
        "threading_patched=%s Thread_impl=%s local_impl=%s\n" % (
            len(events),
            getattr(dds, "_dds_configured_max_threads", "?"),
            getattr(pool, "_max_workers", "?"),
            len(live) if live is not None else "?",
            len(idents),
            _OS_TID_KIND,
            len(tids),
            len(contexts),
            peak,
            wall,
            cpu,
            _CPU_CLOCK_NAME,
            (cpu / wall) if wall > 0 else 0.0,
            work,
            _concurrency_verdict(len(tids), peak, cpu, wall),
            patched,
            getattr(threading.Thread, "__module__", "?"),
            getattr(threading.local, "__module__", "?"),
        ))


def _concurrency_verdict(os_threads, max_simultaneous, cpu, wall):
    """One-word reading of the numbers, so the log cannot be misread later.

    SERIAL           one kernel thread ran the whole batch - the gevent case.
    SERIAL_INTERVALS several kernel threads exist but no two solve intervals
                     overlapped: real threads that never ran at once, which is
                     what a native call that keeps the GIL looks like.
    BLOCKED          intervals overlap on several threads but barely more than
                     one core of CPU was burned - they were waiting, not solving.
    PARALLEL         several kernel threads overlapping AND several cores busy.
    """
    busy = (cpu / wall) if wall > 0 else 0.0
    if os_threads < 2 or max_simultaneous < 2:
        return "SERIAL" if os_threads < 2 else "SERIAL_INTERVALS"
    if busy < 1.5:
        return "BLOCKED"
    return "PARALLEL"


class DDSolver:

    # Default for dds_mode changed to 1
    # Transport table will be reused if same trump suit and the same or nearly the same cards distribution, deal.first can be different.
    # Always search to find the score. Even when the hand to play has only one card, with possible equivalents, to play.
    # If zero, we not always find the score
    # If 2 transport tables ignore trump

    def __init__(self, dds_mode=1, max_threads=0, verbose=False):
        self.dds_mode = dds_mode
        # max_threads is the size of the solver thread pool.
        # 0 = one thread per CPU core.
        workers = max_threads if (max_threads and max_threads > 0) else (os.cpu_count() or 4)
        self._workers = workers
        self._dds_configured_max_threads = max_threads
        self._pool = _SolverPool(max_workers=workers, thread_name_prefix="dds")
        if verbose:
            sys.stderr.write(f"DDSolver loaded — DDS {self.version()} - dds mode {dds_mode} - {workers} solver threads\n")
        # No-op unless --ddsrecord / BEN_DDS_RECORD asked for a recording. If the
        # entry point already opened one, this just labels it with the DDS build
        # actually in use. See ddsrecorder.py / ddsreplay.py.
        DDSRecorder.configure(dds_version=self.version(), dds_mode=dds_mode,
                              threads=workers)

    def version(self):
        return "3.0.0"

    def calculatepar(self, hand, vuln, print_result=True):
        with ModelTimer.time_call('dds_par'):
            t0 = time.perf_counter()
            result = self._calculatepar_impl(hand, vuln, print_result)
            if DDSRecorder.enabled:
                DDSRecorder.record_par(hand, vuln, result,
                                       (time.perf_counter() - t0) * 1000)
            return result

    def _calculatepar_impl(self, hand, vuln, print_result=True):
        # vulnerable
        # 0: None 1: Both 2: NS 3: EW
        v = 0
        if vuln[0]: v = 2
        if vuln[1]: v = 3
        if vuln[0] and vuln[1]: v = 1

        # calc_all_tables_pbn computes the DD table and (mode != -1) the par score.
        try:
            result = dds3.calc_all_tables_pbn(["N:" + hand], mode=v)
        except Exception as e:
            sys.stderr.write(f"Error calculating par: {e}, Hand {hand.encode('utf-8')}\n")
            raise

        par_results = result.get("par_results") or []
        if not par_results:
            sys.stderr.write(f"{Fore.RED}No par result for hand {hand.encode('utf-8')}{Style.RESET_ALL}\n")
            return None

        par = par_results[0]
        # par_score is a 2-element list of strings, e.g. "NS 420" / "EW -420".
        ns_par = par["par_score"][0]
        ew_par = par["par_score"][1]

        if print_result:
            print("NS score: {}".format(ns_par))
            print("EW score: {}".format(ew_par))

        return int(ns_par.split()[1])

    # Solutions
    #1	Find the maximum number of tricks for the side to play.  Return only one of the optimum cards and its score.
    #2	Find the maximum number of tricks for the side to play.  Return all optimum cards and their scores.
    #3	Return all cards that can be legally played, with their scores in descending order.

    @staticmethod
    def _trick_number(hands_pbn, current_trick):
        """Derive trick number (1-13) from remaining cards in PBN hand."""
        pbn = hands_pbn[0]
        if ':' in pbn:
            pbn = pbn.split(':', 1)[1]
        remaining = sum(1 for c in pbn if c not in '. ')
        return (52 - remaining - len(current_trick)) // 4 + 1

    def solve(self, strain_i, leader_i, current_trick, hands_pbn, solutions, purpose=""):
        """Solve a batch of sampled hands with double-dummy.

        purpose is a short tag describing *why* DDS was invoked (e.g. "play",
        "claimcheck", "lead", "bid"). It is folded into the timing label so the
        MODEL TIMING SUMMARY separates, say, card-play evaluations from claim
        checks at the same trick instead of lumping them into one opaque count.
        """
        trick = self._trick_number(hands_pbn, current_trick)
        label = f'dds_solve_t{trick:02d}_{purpose}' if purpose else f'dds_solve_t{trick:02d}'
        with ModelTimer.time_call(label, items=len(hands_pbn)):
            t0 = time.perf_counter()
            results = self.solve_helper(strain_i, leader_i, current_trick, hands_pbn, solutions)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        if DDSRecorder.enabled:
            DDSRecorder.record_solve(strain_i, leader_i, current_trick, hands_pbn,
                                     solutions, purpose, trick, results, elapsed_ms)

        return results

    def solve_helper(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        card_rank = [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004]

        trump = (strain_i - 1) % 5

        # The current trick is the same for every board in the batch.
        trick_suit = [0, 0, 0]
        trick_rank = [0, 0, 0]
        for i in range(min(3, len(current_trick))):
            trick_suit[i] = current_trick[i] // 13
            trick_rank[i] = 14 - current_trick[i] % 13
        trick_suit = tuple(trick_suit)
        trick_rank = tuple(trick_rank)
        dds_mode = self.dds_mode

        # Solve each board on the thread pool. solve_board_pbn releases the GIL
        # during the DDS search; each pool thread uses its own SolverContext.
        def _solve_one(pbn):
            return dds3.solve_board_pbn(
                pbn,
                trump=trump,
                first=leader_i,
                current_trick_suit=trick_suit,
                current_trick_rank=trick_rank,
                target=-1,
                solutions=solutions,
                mode=dds_mode,
                context=_thread_context(),
            )

        # Diagnostic only (BEN_DDS_CONCURRENCY_PROBE). Unset, task is _solve_one
        # and this batch behaves exactly as before. Set, every board is timed
        # inside the worker that runs it, so the report can tell genuine OS
        # thread overlap apart from greenlets taking turns on one thread. The
        # probe observes only: it changes no argument, no result and no pool.
        events = [] if _dds_probe_enabled() else None
        task = _solve_one
        if events is not None:
            def task(pbn):
                start = time.perf_counter()
                cpu0 = _CPU_CLOCK()
                who = (_os_thread_id(), threading.get_ident(), id(_thread_context()))
                try:
                    return _solve_one(pbn)
                finally:
                    events.append(who + (start, time.perf_counter(), cpu0, _CPU_CLOCK()))

        try:
            solved = list(self._pool.map(task, hands_pbn))
        except Exception as e:
            print(f"{Fore.RED}DDS error: {e} {hands_pbn[0].encode('utf-8')} {current_trick} {leader_i}{Style.RESET_ALL}")
            return None
        finally:
            if events is not None:
                _dds_concurrency_report(self, events)

        if solutions == 1:
            # Just return the maximum number of the side to play for each sample
            card_results = {}
            card_results["max"] = []
            card_results["min"] = []
            for fut in solved:
                card_results["max"].append(fut["score"][0])
                card_results["min"].append(fut["score"][fut["cards"] - 1])

        else:
            card_results = {}
            for fut in solved:
                for i in range(fut["cards"]):
                    suit_i = fut["suit"][i]
                    card = suit_i * 13 + 14 - fut["rank"][i]
                    if card not in card_results:
                        card_results[card] = []
                    card_results[card].append(fut["score"][i])
                    eq_cards_encoded = fut["equals"][i]
                    for k, rank_code in enumerate(card_rank):
                        if rank_code & eq_cards_encoded > 0:
                            eq_card = suit_i * 13 + k
                            if eq_card not in card_results:
                                card_results[eq_card] = []
                            card_results[eq_card].append(fut["score"][i])

        return card_results


    def expected_tricks_dds(self, card_results):
        return {card:round((sum(values)/len(values)),2) for card, values in card_results.items()}

    def expected_tricks_dds_probability(self, card_results, probabilities_list : List[float]):
        # Convert to plain Python list to avoid numpy scalar overhead
        probs = [float(p) for p in probabilities_list]
        return {card: round(sum(p*res for p, res in zip(probs, result_list)),2) for card, result_list in card_results.items()}

    def p_made_target(self, tricks_needed):

        def fun(card_results):
            return {card:round(sum(1 for x in values if x >= tricks_needed)/len(values),3) for card, values in card_results.items()}
        return fun

    def print_dd_results(self, dd_solved, print_result=True, xcards=False):
        print("DD Result\n".join(
            f"{Card.from_code(int(k))}: [{', '.join(f'{x:>2}' for x in v[:20])}{' ]' if len(v) <= 20 else ' ...]'}"
            for k, v in dd_solved.items()
        ))

        # Create a new dictionary to store sorted counts for each key
        sorted_counts_dict = {}

        # Loop through the dictionary and process each key-value pair
        for key, array in dd_solved.items():
            # Use Counter to count the occurrences of each element
            element_count = Counter(array)

            # Sort the counts by frequency in descending order
            sorted_counts = sorted(element_count.items(), key=lambda x: x[1], reverse=True)

            # Store the sorted result in the new dictionary
            sorted_counts_dict[key] = sorted_counts

        # Print the sorted counts for each key
        for key, sorted_counts in sorted_counts_dict.items():
            print(f"Sorted counts for {Card.from_code(int(key), xcards)} DD:")
            for value, count in sorted_counts:
                print(f"  Tricks: {value}, Count: {count}")
