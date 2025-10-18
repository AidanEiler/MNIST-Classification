import numpy as np
import sys
import time
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent.parent.parent)
)  # this piece of code I need so my training file can easily find where src is

# self-made files
from src.utils.data_loader import load, split
from src.models.naive_bayes import NaiveBayes


def calculate_accuracy(PREDICTIONS, TRUE_LABELS):
    """figure out what percentage we got right"""
    correct = np.sum(PREDICTIONS == TRUE_LABELS)
    total = len(TRUE_LABELS)
    return correct / total


def main():
    print("=" * 60)
    print("MNIST Classification - Naive Bayes")
    print("=" * 60)

    total_start = time.time()

    # load images from disk
    print("\nstep 1: loading mnist data...")
    load_start = time.time()
    images, labels = load(DATA_DIR="MNIST", FLATTEN=True, NORMALIZE=True)
    load_time = time.time() - load_start
    print(f"data loading took {load_time:.2f} seconds")

    # shuffle and split into train/test
    print("\nstep 2: splitting into train/test sets...")
    split_start = time.time()
    x_train, x_test, y_train, y_test = split(images, labels, TEST_SIZE=0.2, SEED=42)
    split_time = time.time() - split_start
    print(f"data splitting took {split_time:.2f} seconds")

    print("\nstep 3: training and testing naive bayes...")
    print("=" * 60)

    # train the model (calculates probabilities)
    fit_start = time.time()
    nb = NaiveBayes()
    nb.fit(x_train, y_train)
    fit_time = time.time() - fit_start

    # run predictions
    print(f"\nmaking predictions on {len(x_test)} test images...")
    predict_start = time.time()
    predictions = nb.predict(x_test)
    predict_time = time.time() - predict_start

    # calculate accuracy
    accuracy = calculate_accuracy(predictions, y_test)

    total_time = time.time() - total_start

    # show final results
    print("\n" + "=" * 60)
    print("summary of results:")
    print("=" * 60)
    print(f"fit time: {fit_time:.2f} seconds")
    print(f"prediction time: {predict_time:.2f} seconds")
    print(f"total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n" + "=" * 60)
    print("naive bayes training and testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()