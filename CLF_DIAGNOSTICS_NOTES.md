# CLF diagnostics and graceful non-adaptive baseline

This incremental patch does not change the physical or auxiliary control laws.

It adds:

1. Graceful non-adaptive conservative-domain failure:
   the four sensing constraint values are checked before evaluating any
   logarithmic barrier. If an enabled conservative `h_c` reaches the tiny
   positive log-domain floor, the affected follower switches to the existing
   stationkeeping fallback instead of raising `ValueError`.

2. Detailed physical CLF histories:
   - `W`;
   - `alpha(W)`;
   - total drift `a`;
   - `zeta^T nu`;
   - parent/interconnection term `chi`;
   - velocity/backstepping drift term;
   - exact best actuator contribution `min b^T f`;
   - minimum modeled CLF derivative;
   - hard-CLF residual
         a + alpha(W) + min b^T f.

   The positive part of the final residual equals `delta_req`.

3. New figures:
   - `clf_value_decay`;
   - `clf_feasibility_balance`;
   - `clf_drift_components`;
   - `clf_peak_detail`, automatically centered on the largest
     actuator-required slack spike.

4. Console output for the exact decomposition at the worst hard-CLF sample.

5. With `--save`, `clf_diagnostics.csv` is exported for offline analysis.

6. The default practical adaptive-domain margin ratio is now `0.1`, matching
   the value selected in the experiments.
