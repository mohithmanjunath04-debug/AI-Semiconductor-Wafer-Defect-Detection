print("✅ LOADED THE CORRECT GRADCAM FILE")
print("🔥 NEW GRADCAM FILE LOADED")
from seaborn import heatmap
import tensorflow as tf
import numpy as np
import cv2


def generate_gradcam(model, img_array, last_conv_layer_name):
    print("🔥 generate_gradcam() is running...")

    # Build model
    _ = model(img_array)

    # Create intermediate model
    grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer(last_conv_layer_name).output,
        model.outputs[0]
    ]
)
    print("Grad model created successfully")
    print("Last conv layer:", last_conv_layer_name)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradient calculation failed.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        pooled_grads * conv_outputs,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8
    print("Heatmap Min:", tf.reduce_min(heatmap).numpy())
    print("Heatmap Max:", tf.reduce_max(heatmap).numpy())
    print("Heatmap Shape:", heatmap.shape)
    return heatmap.numpy()


def overlay_heatmap(heatmap, wafer):

    heatmap = cv2.resize(
        heatmap,
        (wafer.shape[1], wafer.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_TURBO
    )

    wafer_rgb = cv2.cvtColor(
        np.uint8(wafer * 255),
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        wafer_rgb,
        0.65,
        heatmap,
        0.55,
        0
    )

    return overlay