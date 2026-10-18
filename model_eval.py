"""Evaluate the existing six-class model on the folder validation subset.

This is validation accuracy, not an independent held-out test result.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    probabilities = model.predict(generator, verbose=1)
    predicted = np.argmax(probabilities, axis=1)
    actual = generator.classes
    accuracy = accuracy_score(actual, predicted)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(classification_report(
        actual, predicted, labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0
    ))

    matrix = confusion_matrix(actual, predicted, labels=range(len(CLASS_NAMES)))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Folder validation subset | accuracy {accuracy:.1%}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
