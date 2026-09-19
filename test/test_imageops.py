"""Geometry tests.

These guard the failure that is hardest to notice: preprocessing that is
subtly different from what training used. It never raises -- accuracy just
quietly drops, and it looks like a bad model.
"""

import numpy as np
import pytest

from digitood.constants import IMAGE_SIZE
from digitood.imageops import center_by_mass, fit_to_mnist_frame, ink_bbox, normalize


def test_ink_bbox_finds_the_drawn_region():
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[10:15, 6:9] = 1.0
    assert ink_bbox(arr) == (6, 10, 9, 15)


def test_ink_bbox_is_none_for_a_blank_image():
    assert ink_bbox(np.zeros((28, 28), dtype=np.float32)) is None


def test_blank_input_survives_framing():
    # A user pressing Classify on an empty canvas must not crash the service.
    out = fit_to_mnist_frame(np.zeros((28, 28), dtype=np.float32))
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert out.sum() == 0


def test_framing_is_translation_invariant():
    """The same shape in two corners must produce the same framed output.

    This is the property the whole preprocessing pipeline exists to provide:
    where you draw on the canvas must not change the prediction.
    """
    a = np.zeros((60, 60), dtype=np.float32)
    a[4:20, 6:14] = 1.0
    b = np.zeros((60, 60), dtype=np.float32)
    b[38:54, 44:52] = 1.0
    assert np.allclose(fit_to_mnist_frame(a), fit_to_mnist_frame(b), atol=1e-5)


def test_framing_preserves_aspect_ratio():
    """A tall thin stroke (a '1') must not be stretched into a square."""
    arr = np.zeros((60, 60), dtype=np.float32)
    arr[10:50, 28:32] = 1.0
    bbox = ink_bbox(fit_to_mnist_frame(arr))
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    assert height > width * 2


def test_center_by_mass_shifts_the_centroid_to_the_middle():
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[2:6, 2:6] = 1.0
    out = center_by_mass(arr)
    ys, xs = np.mgrid[0:28, 0:28]
    cy = (ys * out).sum() / out.sum()
    cx = (xs * out).sum() / out.sum()
    assert abs(cy - 13.5) <= 1.0 and abs(cx - 13.5) <= 1.0


def test_center_by_mass_clips_rather_than_wraps():
    """A large shift must not wrap ink around to the opposite edge."""
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[0:3, 0:3] = 1.0
    out = center_by_mass(arr)
    assert out[-3:, -3:].sum() == 0


def test_normalize_matches_mnist_statistics():
    out = normalize(np.full((28, 28), 0.1307, dtype=np.float32))
    assert np.allclose(out, 0.0, atol=1e-5)


def test_framing_is_stable_under_repetition():
    """fit(fit(x)) must equal fit(x) to within resampling noise.

    This is the property that makes it safe to frame at serving time. It is
    not free: framing crops and rescales, so each application resamples. When
    training skipped framing and serving applied it, the model was trained on
    one distribution and served another -- acceptance of real digits fell from
    96% to 45% while the offline metrics still read 0.77% false rejects.

    The project now frames both sides. This test guards the assumption that
    doing so converges rather than drifting.
    """
    rng = np.random.default_rng(4)
    arr = np.zeros((40, 40), dtype=np.float32)
    arr[8:30, 14:19] = 1.0
    arr[20:24, 14:32] = 1.0
    arr += rng.uniform(0, 0.04, arr.shape).astype(np.float32)

    once = fit_to_mnist_frame(np.clip(arr, 0, 1))
    twice = fit_to_mnist_frame(once)
    assert np.abs(twice - once).mean() < 0.02


def test_framing_keeps_the_digit_inside_the_frame():
    """No ink may be pushed outside the 28x28 field by centring."""
    arr = np.zeros((50, 50), dtype=np.float32)
    arr[2:48, 2:8] = 1.0
    out = fit_to_mnist_frame(arr)
    assert out[0, :].sum() == 0 and out[-1, :].sum() == 0
    assert out.sum() > 0
