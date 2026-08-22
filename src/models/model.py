"""Model architectures.

Two are provided so the MLflow experiment has something to compare:

* `baseline_cnn`  - the small from-scratch CNN the assignment asks for as a
                    baseline.
* `mobilenetv2`   - transfer learning on a frozen ImageNet backbone, used as
                    the production model.

Both models embed their own normalisation layer. That is deliberate: the
serving code then only has to resize an image to 224x224, which removes any
chance of train/serve preprocessing skew.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ARCHITECTURES = ("baseline_cnn", "mobilenetv2")


def build_augmenter(
    horizontal_flip: bool = True,
    rotation: float = 0.1,
    zoom: float = 0.2,
    translation: float = 0.1,
    seed: int = 42,
) -> keras.Sequential:
    """Data augmentation applied to the training split only."""
    blocks: list[layers.Layer] = []
    if horizontal_flip:
        blocks.append(layers.RandomFlip("horizontal", seed=seed))
    if rotation:
        blocks.append(layers.RandomRotation(rotation, seed=seed))
    if zoom:
        blocks.append(layers.RandomZoom(zoom, seed=seed))
    if translation:
        blocks.append(layers.RandomTranslation(translation, translation, seed=seed))
    return keras.Sequential(blocks or [layers.Identity()], name="augmentation")


def build_baseline_cnn(input_shape: tuple[int, int, int] = (224, 224, 3)) -> keras.Model:
    """Small CNN trained from scratch - the required baseline model."""
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="baseline_cnn",
    )


def build_mobilenetv2(input_shape: tuple[int, int, int] = (224, 224, 3)) -> keras.Model:
    """MobileNetV2 feature extractor (frozen) + a small classification head."""
    try:
        base = keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    except Exception as exc:  # no network / weight cache available
        raise RuntimeError(
            "Could not fetch the ImageNet weights for MobileNetV2. Run with "
            "ARCHITECTURE=baseline_cnn for an offline, from-scratch baseline, "
            f"or restore network access. Original error: {exc}"
        ) from exc
    base.trainable = False
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Rescaling(1.0 / 127.5, offset=-1.0),  # MobileNetV2 expects [-1, 1]
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="mobilenetv2_transfer",
    )


def build_model(
    architecture: str = "mobilenetv2",
    image_size: int = 224,
    learning_rate: float = 1e-4,
) -> keras.Model:
    """Build and compile the requested architecture."""
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {architecture!r}, expected one of {ARCHITECTURES}"
        )
    input_shape = (image_size, image_size, 3)
    model = (
        build_baseline_cnn(input_shape)
        if architecture == "baseline_cnn"
        else build_mobilenetv2(input_shape)
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_parameters(model: keras.Model) -> dict[str, int]:
    trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    non_trainable = int(sum(tf.size(w).numpy() for w in model.non_trainable_weights))
    return {
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "total_params": trainable + non_trainable,
    }
