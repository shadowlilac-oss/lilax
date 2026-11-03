"""Alternative DeepSeek model definition with batch-split schedule.

The model logic and optimizations are very explicit in this implementation.
Weights are explicitly pre-fetched and gathered in the forward pass and gradients
are explicitly reduced and post-scattered in the backward pass. Optimization
barriers are used to enforce ordering of both large blocks of operations (e.g.
attention, dispatch, etc) and individual operations (e.g. AG+gather within
dispatch). In order to control remat, residuals from the forward pass are
explicitly stored and passed to the backward pass in a custom VJP over the
entire layer scan. The backward pass comprises of remat/bwd functions for each
forward pass function, with relevant residuals passed between them.
"""
