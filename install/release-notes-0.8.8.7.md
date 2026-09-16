## Serving

- **Concurrent API requests no longer fail with "Attempting to capture an
  EagerTensor without building a function".** `gameapi.py` runs every request as
  a gevent greenlet on one OS thread, but TensorFlow keeps its eager/graph
  execution flag per OS thread, where gevent cannot reach it. While a
  `tf.function` traces, that flag says graph mode, and TensorFlow yields on every
  graph op it creates - so any greenlet that ran in one of those windows failed.
  All TensorFlow entry points now go through `tf_lock` in the new
  `src/nn/tf_guard.py`. Outside gevent the lock is a no-op, so real threads keep
  running inference in parallel.

## API

- **Local traffic is exempt from all rate limits, not just some of it.** Only
  `/play` and `/bids` named the exemption, so `/bid` and `/lead` counted against
  20000 per day even from `127.0.0.1`. A robot match long enough to reach the cap
  got HTTP 429 and lost the board. Limits for external callers are unchanged.
- **`--nolimit` works.** It was parsed into a variable that nothing ever read.
- **The private-network exemption covers all of 172.16.0.0/12** rather than just
  `172.16.` and `172.17.`, so any Docker network qualifies, not only the default
  bridge.

## Bidding

- **Fixed a rescue bid picking the most optimistic trick count instead of the
  most likely one.** `trick_score` was re-initialised inside the trick loop,
  making the comparison always true. The rescue now also requires
  `min_samples_for_rescue` samples (config, `[bidding]`, default 15): it is a
  majority vote across samples, which is noise at n=2.

## Tools

- **`--facit` prints the Pavlicek recommended auction** alongside the one BEN
  produced.

## Packaging

- **New `BENAPI` release package.** API-only: `gameapi.exe` plus the models,
  configs, `BBA\CC` card files, `bin\` native libraries and `nn\` sources it
  needs - no game server, no appserver, no GUI, no table manager client. It
  ships the same system selection as `MvsM`, plus `config\default_api.conf`,
  which is what `gameapi.exe` loads when started without `--config`.

## Native libraries

- **SuitC updated from 0.9.0.8 to 0.9.0.11** on Windows, macOS and Linux.
