"""
jacobson.py
===========
The classical baseline your model is competing against.

This implements the standard TCP retransmission timer from RFC 6298, which is
Jacobson's 1988 algorithm plus Karn's refinements.  Every mainstream TCP stack
uses some version of this.

THE ALGORITHM (RFC 6298, section 2)
-----------------------------------
Two state variables are maintained:

    SRTT    "smoothed round-trip time"  - the running estimate of RTT
    RTTVAR  "round-trip time variation" - the running estimate of how much
                                          RTT bounces around

On the FIRST measurement R:
    SRTT   = R
    RTTVAR = R / 2

On EVERY SUBSEQUENT measurement R':
    RTTVAR = (1 - beta) * RTTVAR + beta * |SRTT - R'|
    SRTT   = (1 - alpha) * SRTT  + alpha * R'

    ^ NOTE THE ORDER.  RTTVAR is updated FIRST, using the OLD SRTT.  Getting
      this backwards is the single most common implementation bug here, and
      it is exactly the kind of detail a viva question lands on.

And in both cases:
    RTO = SRTT + max(G, K * RTTVAR)

with the standard constants alpha = 1/8, beta = 1/4, K = 4, and G = the clock
granularity.

WHY THIS IS A *REACTIVE* ESTIMATOR (this is your project's whole argument)
-------------------------------------------------------------------------
Look at the SRTT update.  It is an exponentially weighted moving average with
alpha = 1/8, which means each new measurement only moves the estimate by 1/8 of
the gap.  That is deliberately sluggish -- it smooths out noise.  But it also
means that when the network genuinely changes (congestion starts, a route
flips), SRTT needs many samples to catch up.  During that lag the RTO is wrong:
too low right after RTT rises (spurious retransmissions) and too high right
after RTT falls (wasted waiting).

A sequence model can, in principle, see the *pattern* leading into a congestion
event and move its estimate before the averaging would have.  That is the
hypothesis your project tests.

ON THE 1-SECOND FLOOR
---------------------
RFC 6298 says RTO should be clamped to a minimum of 1 second.  For a comparison
at millisecond scale that floor would swamp everything -- every method would
just output 1000 ms and the comparison would be meaningless.  So `min_rto` below
defaults to 0 (floor disabled) for the experiment.  Mention this in your report:
it is a legitimate, deliberate deviation for experimental purposes, not an
oversight.  Being upfront about it is worth more than hoping nobody notices.

ON KARN'S ALGORITHM
-------------------
Karn's algorithm has two parts:
  (a) do not take an RTT sample from a retransmitted segment (the ACK is
      ambiguous -- you cannot tell which copy it acknowledges), and
  (b) on timeout, back off the RTO exponentially.

Part (a) is vacuous for our dataset: ping data has no retransmissions, so every
sample is unambiguous.  Part (b) is implemented below as an option.  Say this
plainly in the report rather than claiming a fuller Karn implementation than you
actually have.
"""

import numpy as np


ALPHA = 1.0 / 8.0   # SRTT smoothing factor
BETA = 1.0 / 4.0    # RTTVAR smoothing factor
K = 4.0             # how many variation-widths of safety margin to add


def jacobson_rto(rtt_series, alpha=ALPHA, beta=BETA, k=K,
                 granularity=1.0, min_rto=0.0, max_rto=60000.0):
    """Run the RFC 6298 estimator over a series of RTT measurements.

    Parameters
    ----------
    rtt_series : 1-D array of measured RTTs, in milliseconds, in time order.
    granularity : G, the clock granularity in ms.
    min_rto, max_rto : clamps in ms.  min_rto=0 disables the RFC's 1 s floor
        (see the module docstring for why).

    Returns
    -------
    dict with three arrays, each the same length as the input:
        'srtt'   - the smoothed RTT estimate at each step
        'rttvar' - the variation estimate at each step
        'rto'    - the timeout that would have been used at each step

    IMPORTANT: entry i of each array is the estimate computed AFTER observing
    sample i.  So the value that would actually be *used as a prediction* for
    sample i+1 is entry i.  The evaluation script shifts by one accordingly --
    otherwise you would be crediting the algorithm with knowing the answer
    before it saw it, which is the classic look-ahead bug.
    """
    rtt_series = np.asarray(rtt_series, dtype=float)
    n = len(rtt_series)

    srtt_out = np.zeros(n)
    rttvar_out = np.zeros(n)
    rto_out = np.zeros(n)

    srtt = 0.0
    rttvar = 0.0

    for i, r in enumerate(rtt_series):
        if i == 0:
            # First measurement: initialise directly from it.
            srtt = r
            rttvar = r / 2.0
        else:
            # Order matters: RTTVAR first, using the OLD srtt.
            rttvar = (1 - beta) * rttvar + beta * abs(srtt - r)
            srtt = (1 - alpha) * srtt + alpha * r

        rto = srtt + max(granularity, k * rttvar)
        rto = float(np.clip(rto, min_rto, max_rto))

        srtt_out[i] = srtt
        rttvar_out[i] = rttvar
        rto_out[i] = rto

    return {"srtt": srtt_out, "rttvar": rttvar_out, "rto": rto_out}


def karn_backoff(rto, n_consecutive_timeouts, max_rto=60000.0):
    """Karn's exponential backoff: double the RTO on each consecutive timeout.

    Not used in the main pipeline (our ping data contains no timeouts), but
    included so you can describe and demonstrate it if asked.
    """
    return float(min(rto * (2 ** n_consecutive_timeouts), max_rto))


if __name__ == "__main__":
    # Tiny self-check you can run to convince yourself the recursion is right.
    demo = np.array([100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 200.0])
    out = jacobson_rto(demo)
    print("RTT    SRTT    RTTVAR   RTO")
    for r, s, v, o in zip(demo, out["srtt"], out["rttvar"], out["rto"]):
        print(f"{r:6.1f} {s:7.2f} {v:7.2f} {o:8.2f}")
    print("\nNotice how slowly SRTT climbs after the jump from 100 to 200.")
    print("That lag is exactly what your model is trying to beat.")
