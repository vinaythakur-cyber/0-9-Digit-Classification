"""Every negative generator must emit a valid, non-degenerate image."""

import numpy as np
import pytest

from digitood.constants import IMAGE_SIZE
from digitood.data.glyphs import CHARSET, render_glyph
from digitood.data.negatives import (
    SYNTHETIC_GENERATORS, patch_shuffle, sample_synthetic,
)


@pytest.mark.parametrize("name", sorted(SYNTHETIC_GENERATORS))
def test_generator_contract(name):
    fn = SYNTHETIC_GENERATORS[name][0]
    rng = np.random.default_rng(0)
    for _ in range(20):
        arr = fn(rng)
        assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert arr.dtype == np.float32
        assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0
        assert np.isfinite(arr).all()


def test_generators_are_reproducible_from_a_seed():
    a, _ = sample_synthetic(np.random.default_rng(7))
    b, _ = sample_synthetic(np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_generators_are_not_constant():
    """Different seeds must give different images, or the negative set is one
    image repeated and the model will trivially memorise it."""
    a, _ = sample_synthetic(np.random.default_rng(1))
    b, _ = sample_synthetic(np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_patch_shuffle_preserves_pixels_but_not_layout():
    rng = np.random.default_rng(3)
    digit = rng.random((IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    out = patch_shuffle(digit, rng)
    # Same multiset of pixel values -- the point of this family is that no
    # intensity statistic can separate it from a real digit.
    assert np.allclose(np.sort(out.ravel()), np.sort(digit.ravel()))
    assert not np.array_equal(out, digit)


def test_charset_excludes_digit_lookalikes():
    """'O' and 'l' are visually identical to 0 and 1; labelling them
    'unclassifiable' would teach a distinction that is not in the pixels."""
    for bad in "OoIl":
        assert bad not in CHARSET


def test_charset_contains_no_actual_digits():
    assert not any(c.isdigit() for c in CHARSET)


def test_glyph_rendering_produces_ink():
    rng = np.random.default_rng(11)
    for _ in range(15):
        arr, char = render_glyph(rng)
        assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert arr.max() > 0.3, f"glyph {char!r} rendered blank"
