from trapred.data.splits import assign_split


def test_train_future_does_not_overlap_val_history():
    fps = 10.0
    n_frames = 3000
    t_in, t_out = 3.0, 5.0
    train_last, val_first = [], []
    for t_last in range(int(t_in * fps), n_frames - int(t_out * fps), 10):
        s = assign_split(
            t_last, n_frames, fps=fps, t_input_s=t_in, t_horizon_s=t_out
        )
        if s == "train":
            train_last.append(t_last)
        elif s == "val":
            val_first.append(t_last)
    assert train_last and val_first
    train_future_end = max(train_last) / fps + t_out
    val_hist_start = min(val_first) / fps - t_in
    assert val_hist_start >= train_future_end - 1e-6


def test_discard_covers_the_buffer():
    fps = 10.0
    n_frames = 1000
    labels = [
        assign_split(t, n_frames, fps=fps, t_input_s=3.0, t_horizon_s=5.0)
        for t in range(30, 950, 5)
    ]
    assert "discard" in labels
    assert labels.count("train") > 0
    assert labels.count("val") > 0
    assert labels.count("test") > 0
