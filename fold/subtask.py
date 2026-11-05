"""Subtask annotations for fold episodes (the FineART idea, arXiv 2609.36416, adapted to state-based BC).

FineART labels every demo with contiguous, non-overlapping subtask segments that each have a caption. A "System 2"
head predicts the current subtask from the observation, and a "System 1" action policy is conditioned on that
subtask. Here, every fold demo comes from a planner with known stage boundaries (reach / descend / grasp / transport
/ release / retreat; see fold/retarget.py), so the segments are exact and free. The transport segment is split
into lift / carry / lower at the same 25% / 75% points the planner uses for its clearance profile.
"""
import numpy as np

from fold.retarget import N_REACH, DWELL

CAPTIONS = [
    "reach above both near corners",
    "descend onto the near corners",
    "close both hands to pinch the corners",
    "lift the near half off the table",
    "carry the near half over the far half",
    "lower the near corners onto the far corners",
    "open both hands to release",
    "retreat up and away from the cloth",
]
NSUB = len(CAPTIONS)


def labels(T, k_grasp, k_release):
    """Per-step subtask id (0..7) for an episode of T steps with the planner's grasp/release steps."""
    y = np.full(T, 7)
    t = np.arange(T)
    t_lower = k_release - DWELL                    # transport ends where the release ramp starts
    al = (t - k_grasp) / max(1, t_lower - k_grasp)
    y[t < k_release] = 6
    y[t < t_lower] = np.where(al[t < t_lower] < 0.75, 4, 5)
    y[(t < t_lower) & (al < 0.25)] = 3
    y[t < k_grasp] = 2
    y[t < N_REACH] = 1
    y[t < N_REACH // 2] = 0
    return y
