#!/usr/bin/env python3
"""
Fanorona AlphaZero ResNet – mifanandrify tanteraka amin'ny
lib/logic/ai/nn_encoding.dart sy alphazero_ai.dart

- Input  : (B, 5, 9, 7)
- Policy : (B, 1080)
- Value  : (B, 1)
- Export : fanorona_model.tflite (float32, no quantization by default)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ====================== Constants (mifanandrify amin'ny Dart) ======================
ROWS = 5
COLS = 9
NUM_PLANES = 7
NUM_DIRECTIONS = 8
NUM_MOVE_TYPES = 3          # none, approach, withdrawal
MAX_ACTIONS = ROWS * COLS * NUM_DIRECTIONS * NUM_MOVE_TYPES  # 1080

# ====================== Residual Block ======================
def residual_block(x, filters: int, name: str):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x

# ====================== Model Builder ======================
def build_fanorona_resnet(
    num_res_blocks: int = 8,
    filters: int = 128,
    policy_filters: int = 32,
    value_hidden: int = 256,
) -> keras.Model:
    """
    AlphaZero-style dual-head network for Fanorona.
    Small enough for mobile (tflite) yet strong enough for good play.
    """
    inputs = keras.Input(shape=(ROWS, COLS, NUM_PLANES), name="board")

    # Stem
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Residual tower
    for i in range(num_res_blocks):
        x = residual_block(x, filters, name=f"res{i}")

    # ---------- Policy head ----------
    p = layers.Conv2D(policy_filters, 1, use_bias=False, name="policy_conv")(x)
    p = layers.BatchNormalization(name="policy_bn")(p)
    p = layers.ReLU(name="policy_relu")(p)
    p = layers.Flatten(name="policy_flat")(p)
    policy = layers.Dense(MAX_ACTIONS, activation="softmax", name="policy")(p)

    # ---------- Value head ----------
    v = layers.Conv2D(1, 1, use_bias=False, name="value_conv")(x)
    v = layers.BatchNormalization(name="value_bn")(v)
    v = layers.ReLU(name="value_relu")(v)
    v = layers.Flatten(name="value_flat")(v)
    v = layers.Dense(value_hidden, activation="relu", name="value_fc")(v)
    value = layers.Dense(1, activation="tanh", name="value")(v)

    model = keras.Model(inputs=inputs, outputs=[policy, value], name="FanoronaResNet")
    return model

# ====================== Loss & Compile ======================
def compile_model(model: keras.Model, learning_rate: float = 1e-3):
    """
    Standard AlphaZero losses:
    - Policy : categorical cross-entropy (with soft targets from MCTS)
    - Value  : mean squared error
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={
            "policy": keras.losses.CategoricalCrossentropy(),
            "value": keras.losses.MeanSquaredError(),
        },
        loss_weights={"policy": 1.0, "value": 1.0},
        metrics={
            "policy": [keras.metrics.CategoricalAccuracy(name="policy_acc")],
            "value": [keras.metrics.MeanAbsoluteError(name="value_mae")],
        },
    )
    return model

# ====================== Dummy data (for smoke test / warm-up) ======================
def make_dummy_batch(batch_size: int = 16):
    """Generate random legal-looking tensors (useful for shape checking)."""
    x = np.random.rand(batch_size, ROWS, COLS, NUM_PLANES).astype(np.float32)
    # Normalize planes roughly like real encoding
    x[..., 0:3] = (x[..., 0:3] > 0.5).astype(np.float32)  # own / opp / empty
    x[..., 3] = ((np.arange(ROWS)[:, None] + np.arange(COLS)) % 2 == 0).astype(np.float32)
    policy = np.random.dirichlet(np.ones(MAX_ACTIONS), size=batch_size).astype(np.float32)
    value = np.random.uniform(-1, 1, size=(batch_size, 1)).astype(np.float32)
    return x, {"policy": policy, "value": value}

# ====================== Export to TFLite ======================
def export_tflite(model: keras.Model, path: str = "fanorona_model.tflite", quantize: bool = False):
    """
    Convert Keras model → TensorFlow Lite.
    The resulting .tflite is directly usable by tflite_flutter
    (policy lastDim == 1080, value lastDim == 1).
    """
    # Ensure concrete function with fixed batch=1 for mobile
    @tf.function(input_signature=[tf.TensorSpec([1, ROWS, COLS, NUM_PLANES], tf.float32, name="board")])
    def serving(board):
        policy, value = model(board, training=False)
        return {"policy": policy, "value": value}

    concrete_fn = serving.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT] if quantize else []
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,  # safety
    ]
    tflite_model = converter.convert()

    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"[✓] Saved TFLite model → {path}  ({len(tflite_model)/1024:.1f} KB)")
    return path

# ====================== Main ======================
if __name__ == "__main__":
    print("Building Fanorona ResNet ...")
    model = build_fanorona_resnet(
        num_res_blocks=8,   # 6–10 recommended; 8 is a good balance for mobile
        filters=128,
        policy_filters=32,
        value_hidden=256,
    )
    model.summary()

    compile_model(model, learning_rate=1e-3)

    # Smoke test
    x, y = make_dummy_batch(8)
    loss = model.train_on_batch(x, y)
    print(f"Smoke train loss: {loss}")

    # Save full Keras model (optional, for further training)
    model.save("fanorona_resnet.keras")
    print("[✓] Saved Keras model → fanorona_resnet.keras")

    # Export TFLite (the one the Flutter app loads)
    export_tflite(model, path="fanorona_model.tflite", quantize=False)

    # Quick inference check (same shapes the Dart side expects)
    interpreter = tf.lite.Interpreter(model_path="fanorona_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n=== TFLite I/O shapes (must match Dart) ===")
    print("Input :", input_details[0]["shape"], input_details[0]["dtype"])
    for o in output_details:
        print("Output:", o["name"], o["shape"], o["dtype"])

    # Run one forward pass
    dummy = np.random.rand(1, 5, 9, 7).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()
    for o in output_details:
        out = interpreter.get_tensor(o["index"])
        print(f"  → {o['name']}: shape={out.shape}, min={out.min():.3f}, max={out.max():.3f}")
