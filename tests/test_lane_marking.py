import numpy as np

from trapred.data.lane_marking import (
    DASHED,
    SOLID,
    _classify_signal,
    extract_lane_markings,
    find_adjacent_pairs,
    lane_span_at_x,
)


def test_classify_solid_vs_dashed():
    solid = [220.0] * 200
    dashed = ([220.0] * 8 + [40.0] * 8) * 20
    assert _classify_signal(solid)[0] == SOLID
    assert _classify_signal(dashed)[0] == DASHED


def test_lane_span_and_adjacent_pair():
    upper = np.array([[0, 0], [100, 0], [100, 10], [0, 10]], dtype=float)
    lower = np.array([[0, 12], [100, 12], [100, 22], [0, 22]], dtype=float)
    su = lane_span_at_x(upper, 50)
    sl = lane_span_at_x(lower, 50)
    assert su is not None and sl is not None
    assert su[1] == 10 and sl[0] == 12
    pairs = find_adjacent_pairs([upper, lower], max_gap_px=5, step=5)
    assert pairs == [(0, 1)]


def test_extract_dashed_from_synthetic_image():
    bg = np.zeros((80, 400, 3), dtype=np.uint8)
    bg[:] = 30
    # dashed white line at y=40
    for x0 in range(0, 400, 24):
        bg[38:43, x0:x0 + 10] = 255
    upper = np.array([[0, 10], [399, 10], [399, 38], [0, 38]], dtype=float)
    lower = np.array([[0, 42], [399, 42], [399, 70], [0, 70]], dtype=float)
    marks = extract_lane_markings([upper, lower], bg, pairs=[(0, 1)])
    assert len(marks) == 1
    assert marks[0].marking == DASHED
