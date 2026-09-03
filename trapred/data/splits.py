"""Time-based splits that never leak a window's future into another split's history."""
from __future__ import annotations


def assign_split(
    t_last: int,
    n_frames: int,
    *,
    fps: float,
    t_input_s: float,
    t_horizon_s: float,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> str:
    """Assign a window by its last observed frame.

    A window occupies ``[t_last - t_in, t_last + t_out]``. Train futures
    stop at the train cut; val/test histories start after that cut, so
    no frame is both a train label and a later-split input.
    """
    duration_s = n_frames / fps
    t1 = duration_s * train_frac
    t2 = duration_s * (train_frac + val_frac)
    t_last_s = t_last / fps
    if t_last_s + t_horizon_s <= t1:
        return "train"
    if t_last_s - t_input_s >= t1 and t_last_s + t_horizon_s <= t2:
        return "val"
    if t_last_s - t_input_s >= t2:
        return "test"
    return "discard"
