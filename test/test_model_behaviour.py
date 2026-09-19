"""End-to-end behaviour of the trained model.

Skipped automatically when the checkpoint or the MNIST cache is missing, so
CI stays fast and offline. Run locally after `make train`.

These are the tests that catch the failures unit tests cannot see: the model
loads, the preprocessing matches, and the accept/reject decision is right on
real inputs rather than on synthetic tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from digitood.imageops import normalize

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "digit_ood_cnn.pt"
MNIST_DIR = ROOT / "data" / "MNIST"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists() or not MNIST_DIR.exists(),
    reason="needs a trained checkpoint and the MNIST cache; run `make train` first",
)

N = 200  # digits per check -- enough to be meaningful, fast enough to run often


@pytest.fixture(scope="module")
def predictor():
    from digitood.inference.predictor import Predictor
    return Predictor(CHECKPOINT)


@pytest.fixture(scope="module")
def digits():
    from torchvision.datasets import MNIST
    ds = MNIST(str(ROOT / "data"), train=False, download=False)
    return ds.data.numpy()[:N].astype(np.float32) / 255.0, ds.targets.numpy()[:N]


def _accept_rate(predictor, images) -> float:
    batch = np.stack([normalize(img) for img in images])
    return float(np.mean([p.is_digit for p in predictor.predict_batch(batch)]))


def _blur(img: np.ndarray, radius: float) -> np.ndarray:
    from PIL import ImageFilter
    pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)),
                      dtype=np.float32) / 255.0


def _as_canvas_drawing(img: np.ndarray, size: int = 280) -> Image.Image:
    """Turn a 28x28 digit into something shaped like a real canvas drawing.

    Not a plain LANCZOS upscale. Enlarging a 28px image 10x invents no detail
    -- it just produces blurry interpolation, which is a far harsher input
    than anything the app actually receives, and testing against it measures
    the resampler rather than the model.

    A browser canvas draws a crisp vector path and rasterises it with
    antialiasing. Nearest-neighbour upscaling plus a re-threshold reproduces
    the crisp path; the blur reproduces the antialiased edge.
    """
    from PIL import ImageFilter
    pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    big = np.asarray(pil.resize((size, size), Image.Resampling.NEAREST),
                     dtype=np.float32) / 255.0
    crisp = Image.fromarray(((big > 0.5) * 255).astype(np.uint8), mode="L")
    return crisp.filter(ImageFilter.GaussianBlur(size / 80))


def test_accepts_clean_digits(predictor, digits):
    images, _ = digits
    assert _accept_rate(predictor, images) >= 0.95


def test_classifies_accepted_digits_correctly(predictor, digits):
    images, labels = digits
    batch = np.stack([normalize(img) for img in images])
    preds = predictor.predict_batch(batch)
    correct = [p.digit == int(l) for p, l in zip(preds, labels) if p.is_digit]
    assert np.mean(correct) >= 0.95


@pytest.mark.parametrize("radius", [0.4, 0.8, 1.2])
def test_accepts_blurred_digits(predictor, digits, radius):
    """Regression test.

    Several negative families are soft by nature (smooth_blobs, blurred star
    fields) while raw MNIST is crisp. Without blur augmentation on the
    positives, the model learns the shortcut "blurry => not a digit" and then
    refuses real digits that arrive slightly soft -- a phone photo, a rescaled
    upload. That bug rejected 10 of 12 real digits through the HTTP path while
    the offline test reported a 0.77% false-reject rate, because the offline
    test only ever saw pristine 28x28 arrays.
    """
    images, _ = digits
    blurred = [_blur(img, radius) for img in images]
    assert _accept_rate(predictor, blurred) >= 0.85, (
        f"blur radius {radius} caused too many false rejects"
    )


def test_accepts_digits_drawn_at_canvas_resolution(predictor, digits):
    """The real serving path, end to end.

    This is the test that caught the project's worst bug. Training used raw
    MNIST arrays while serving re-framed every input through
    `fit_to_mnist_frame`, and `ink_bbox` counted antialiasing halo as ink --
    so a canvas drawing was framed ~6% smaller than anything the model had
    trained on. Acceptance of genuine digits sat at 45% while the offline
    metrics reported a 0.77% false-reject rate and looked flawless.
    """
    from digitood.inference.preprocess import prepare_image
    images, _ = digits
    batch = np.stack([prepare_image(_as_canvas_drawing(img)) for img in images[:80]])
    rate = float(np.mean([p.is_digit for p in predictor.predict_batch(batch)]))
    assert rate >= 0.90, f"only {rate:.1%} of canvas-drawn digits were accepted"


@pytest.mark.parametrize("family", ["star_field", "gaussian_noise", "scribble",
                                    "texture", "near_blank", "smooth_blobs"])
def test_rejects_synthetic_negatives(predictor, family):
    from digitood.data.negatives import SYNTHETIC_GENERATORS
    rng = np.random.default_rng(99)
    fn = SYNTHETIC_GENERATORS[family][0]
    assert _accept_rate(predictor, [fn(rng) for _ in range(120)]) <= 0.05


def test_rejects_letters(predictor):
    """The hard negatives -- same framing and stroke statistics as digits."""
    from digitood.data.glyphs import render_glyph
    rng = np.random.default_rng(7)
    assert _accept_rate(predictor, [render_glyph(rng)[0] for _ in range(120)]) <= 0.10


def test_prediction_does_not_depend_on_where_you_draw(predictor, digits):
    """Same digit in two corners of a large canvas must give the same answer."""
    from digitood.inference.preprocess import prepare_image
    images, labels = digits
    agree = 0
    for img, label in zip(images[:30], labels[:30]):
        small = Image.fromarray((img * 255).astype(np.uint8), mode="L").resize(
            (90, 90), Image.Resampling.LANCZOS)
        a = Image.new("L", (280, 280), 0); a.paste(small, (10, 10))
        b = Image.new("L", (280, 280), 0); b.paste(small, (170, 160))
        pa = predictor.predict_prepared(prepare_image(a))
        pb = predictor.predict_prepared(prepare_image(b))
        agree += (pa.label == pb.label)
    assert agree >= 28, f"only {agree}/30 gave the same answer in both positions"
