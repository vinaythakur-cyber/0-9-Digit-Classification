"""Input handling: polarity, transparency and arbitrary image sizes."""

import io

import numpy as np
from PIL import Image

from digitood.constants import IMAGE_SIZE, MNIST_MEAN, MNIST_STD
from digitood.inference.preprocess import (
    prepare_array, prepare_bytes, prepare_image,
)


def _denormalize(arr):
    return arr * MNIST_STD + MNIST_MEAN


def test_prepare_array_returns_the_model_input_shape():
    out = prepare_array(np.zeros((100, 100), dtype=np.float32))
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == np.float32


def test_dark_ink_on_white_paper_is_inverted():
    """A photo of a digit on paper is the inverse of MNIST. Without the flip
    the model sees a blob with a digit-shaped hole and is confidently wrong."""
    img = Image.new("L", (100, 100), 255)         # white page
    img.paste(0, (40, 20, 60, 80))                # dark stroke
    out = _denormalize(prepare_image(img))
    assert out.max() > 0.5                         # ink ended up bright
    assert out[:2, :].mean() < 0.2                 # border ended up dark


def test_white_ink_on_black_is_left_alone():
    img = Image.new("L", (100, 100), 0)
    img.paste(255, (40, 20, 60, 80))
    out = _denormalize(prepare_image(img))
    assert out.max() > 0.5
    assert out[:2, :].mean() < 0.2


def test_transparent_canvas_export_uses_the_alpha_channel():
    """An HTML canvas exports RGBA with an alpha=0 background. Converting that
    to grayscale naively can erase the whole drawing."""
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for x in range(40, 60):
        for y in range(20, 80):
            img.putpixel((x, y), (255, 255, 255, 255))
    out = _denormalize(prepare_image(img))
    assert out.max() > 0.5


def test_accepts_a_png_of_an_unusual_size():
    img = Image.new("L", (137, 61), 0)
    img.paste(255, (10, 10, 30, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert prepare_bytes(buf.getvalue()).shape == (IMAGE_SIZE, IMAGE_SIZE)


def test_full_field_texture_is_not_cropped():
    """A star field or noise texture has no meaningful bounding box; cropping
    to its brightest region would invent structure that is not there."""
    rng = np.random.default_rng(0)
    noise = rng.uniform(0.4, 1.0, (64, 64)).astype(np.float32)
    out = prepare_array(noise, frame=False)
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE)
