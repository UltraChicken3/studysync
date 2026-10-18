"""Plot one-vs-rest ROC curves on the folder validation subset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from study_buddy_logic import CLASS_NAMES


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "study_buddy_resnet_final.keras")
    parser.add_argument("--data", type=Path, default=ROOT / "Student Dataset")
    args = parser.parse_args()

    generator = ImageDataGenerator(
        preprocessing_function=preprocess_input, validation_split=0.2
    ).flow_from_directory(
        args.data, target_size=(224, 224), batch_size=32,
        class_mode="categorical", subset="validation", shuffle=False
    )
    expected = {name: i for i, name in enumerate(CLASS_NAMES)}
    if generator.class_indices != expected:
        raise ValueError(f"Folder order differs from model labels: {generator.class_indices}")

    model = tf.keras.models.load_model(args.model, compile=False)
    scores = model.predict(generator, verbose=1)
    targets = label_binarize(generator.classes, classes=range(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(9, 7))
    for index, name in enumerate(CLASS_NAMES):
        RocCurveDisplay.from_predictions(
            targets[:, index], scores[:, index], name=name, ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("One-vs-rest ROC on folder validation subset")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
