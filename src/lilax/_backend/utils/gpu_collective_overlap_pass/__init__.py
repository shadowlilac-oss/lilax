
"""Profile-guided POST_SCHEDULER pass: move async collective starts earlier.

For every async collective in the scheduled HLO, use the PGLE FDO profile to
check whether the ops between start and done hide its latency. If not, move
the start earlier (bounded by data/control dependencies) and pull heavy
compute into the exposed window.

Steps:
  1. Rank 0 computes the new schedule; its serialized module is shared with
     all other ranks through the JAX distributed KV store (entry/exit
     barriers) so every rank compiles the same executable.
  2. Reorder, repeated until nothing changes (at most _MAX_REORDER_ITERS),
     for each computation, innermost first:
       a. Move each under-hidden async start to its earliest legal position,
          relocating the operand producers it depends on.
       b. If a blocker op prevents that, chase it earlier toward heavy
          compute; move an async done ahead of TE EP calls once its latency
          is hidden.
       c. Fill still-exposed windows with heavy compute (best of several
          fill strategies by total exposed time).
       d. In while bodies, hoist async starts to their floor.
     Each iteration ends with schedule.update()/verify().
  3. Finally, hoist host-offload copy starts to their earliest legal position.

Call ``registration.register()`` once before the first ``jax.jit``-compiled function runs.
"""
