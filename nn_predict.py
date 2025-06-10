import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.array(x)
    if x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    else:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
    


# 1. 載入資料
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 正規化
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. 訓練模型
model.fit(x_train, y_train_cat, epochs=20, batch_size=64, validation_split=0.1)

# 5. 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"\nTest accuracy: {test_acc:.4f}")

# 6. 儲存 .h5 模型
model.save("fashion_mnist.h5")

def save_model_architecture(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    arch = []
    for layer in model.layers:
        layer_info = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "config": layer.get_config(),
            "weights": [w.name for w in layer.weights]
        }
        arch.append(layer_info)
    with open(filename, "w") as f:
        json.dump(arch, f, indent=2)

save_model_architecture(model, "model/fashion_mnist.json")

# 8. 儲存權重為 npz
import os

def save_weights_npz(model, filename):
    weights_dict = {}
    for layer in model.layers:
        for w in layer.weights:
            weights_dict[w.name] = w.numpy()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, **weights_dict)

save_weights_npz(model, "model/fashion_mnist.npz")