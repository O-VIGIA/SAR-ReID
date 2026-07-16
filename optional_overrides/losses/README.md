# Optional OpenGait loss overrides

These files preserve the CE and Triplet variants supplied with the research
code. They are not copied by `install_into_opengait.sh`:

- `ce.py` is functionally equivalent to the checked OpenGait implementation.
- `triplet.py` omits two diagnostic fields (`loss_num` and `mean_dist`) from the
  log dictionary but retains the same loss computation.

Only copy them into `OpenGait/opengait/modeling/losses/` if an existing
experiment or checkpoint workflow specifically requires those logging changes.
Review local modifications before overwriting upstream files.

