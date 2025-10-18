import numpy as np
import sys
import time
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent.parent.parent)
)  # this piece of code I need so my training file can easily find where src is

# self-made files
from src.utils.data_loader import load, split
from src.models.knn import KNN


def calculate_accuracy(PREDICTIONS, TRUE_LABELS):
    """figure out what percentage we got right"""
    correct = np.sum(PREDICTIONS == TRUE_LABELS)
    total = len(TRUE_LABELS)
    return correct / total


def main():
    print("=" * 60)
    print("MNIST Classification - K-Nearest Neighbors (KNN)")
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

    # try different k values
    k_values = [1, 3, 5, 7, 9]  # note, bare minimum is 1, 3, 5
    results = {}
    timings = {}

    print("\nstep 3: training and testing knn with different k values...")
    print("=" * 60)

    for k in k_values:
        print(f"\n--- testing knn with k={k} ---")
        k_start = time.time()

        # train the model (just stores the data for knn)
        fit_start = time.time()
        knn = KNN(k=k)
        knn.fit(x_train, y_train)
        fit_time = time.time() - fit_start

        # run predictions
        print(f"making predictions on {len(x_test)} test images...")
        predict_start = time.time()
        predictions = knn.predict(x_test)
        predict_time = time.time() - predict_start

        accuracy = calculate_accuracy(predictions, y_test)

        # save for stats reporting
        k_total_time = time.time() - k_start
        results[k] = accuracy
        timings[k] = {"fit": fit_time, "predict": predict_time, "total": k_total_time}

        print(f"fit time: {fit_time:.2f} seconds")
        print(f"prediction time: {predict_time:.2f} seconds")
        print(f"total time for k={k}: {k_total_time:.2f} seconds")
        print(f"accuracy with k={k}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    total_time = time.time() - total_start

    # show final results
    print("\n" + "=" * 60)
    print("summary of results:")
    print("=" * 60)
    for k in k_values:
        print(f"\nk={k}:")
        print(f"  accuracy: {results[k]:.4f} ({results[k]*100:.2f}%)")
        print(f"  fit time: {timings[k]['fit']:.2f}s")
        print(f"  predict time: {timings[k]['predict']:.2f}s")
        print(f"  total time: {timings[k]['total']:.2f}s")

    # more stats, namely which k did best
    best_k = max(results, key=results.get)
    best_accuracy = results[best_k]
    print(
        f"\nbest k value: {best_k} with accuracy {best_accuracy:.4f} ({best_accuracy*100:.2f}%)"
    )

    print(
        f"\ntotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print("\n" + "=" * 60)
    print("knn training and testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
