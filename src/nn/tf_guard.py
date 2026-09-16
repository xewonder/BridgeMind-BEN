"""Serialise TensorFlow calls when the process runs on gevent greenlets.

TensorFlow keeps its eager/graph execution flag per OS thread, and gevent
cannot patch that away: every greenlet in gameapi.py shares a single OS thread,
so they all share the flag. While a tf.function is being traced the flag says
"graph mode"; if the tracing greenlet yields, whichever greenlet takes over
runs its own eager call with that flag still set and dies with

    RuntimeError: Attempting to capture an EagerTensor without building a
    function.

Holding tf_lock across every TensorFlow entry point keeps two greenlets from
being inside TensorFlow at the same time, so a trace always finishes - and
restores the flag - before anyone else looks at it. Greenlets are cooperative,
so a TensorFlow call only ever gives the lock up at a yield point, which means
this costs nothing in the normal case.

Without gevent the flag really is per thread, there is nothing to guard, and
tf_lock is a no-op so real threads keep running inference in parallel.
"""

from contextlib import nullcontext
from threading import RLock


def _running_on_gevent():
    try:
        from gevent import monkey
    except ImportError:
        return False
    # patch_all() runs before this package is imported, so this is decided once.
    return monkey.is_module_patched('threading')


# RLock, so a guarded call that reaches another guarded call cannot deadlock.
tf_lock = RLock() if _running_on_gevent() else nullcontext()
