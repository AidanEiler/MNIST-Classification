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
from src.utils.visualization import Visualization


def calculate_accuracy(PREDICTIONS, TRUE_LABELS):
    """figure out what percentage we got right"""
    correct = np.sum(PREDICTIONS == TRUE_LABELS)
    total = len(TRUE_LABELS)
    return correct / total


def compute_all_distances(x_train, x_test):
    """
    pre-compute euclidean distances from all test samples to all training samples.
    this is done once and reused for different k values.
    
    args:
        x_train: training images, shape (n_train, 784)
        x_test: test images, shape (n_test, 784)
    
    returns:
        all_distances: array of shape (n_test, n_train) where each row contains
                      distances from one test sample to all training samples
    """
    n_test = len(x_test)
    all_distances = []
    
    for i, test_sample in enumerate(x_test):
        # show progress every 500 samples
        if (i + 1) % 500 == 0:
            print(f"  computing distances for test sample {i + 1}/{n_test}...")
        
        # compute euclidean distance to all training samples
        distances = np.sqrt(np.sum((x_train - test_sample) ** 2, axis=1))
        all_distances.append(distances)
    
    return np.array(all_distances)


def predict_with_k(all_distances, y_train, k):
    """
    make predictions using pre-computed distances and a specific k value.
    
    args:
        all_distances: pre-computed distances, shape (n_test, n_train)
        y_train: training labels
        k: number of nearest neighbors to consider
    
    returns:
        predictions: array of predicted labels
    """
    predictions = []
    
    for distances in all_distances:
        # find k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        
        # majority vote
        prediction = np.bincount(k_nearest_labels).argmax()
        predictions.append(prediction)
    
    return np.array(predictions)


def train():
    """
    train and evaluate knn classifier with multiple k values.
    optimized to compute distances once and reuse for all k values.
    returns results dictionary for the best k value.
    
    returns:
        dict: {
            'accuracy': final test accuracy as percentage,
            'train_time': time spent fitting,
            'total_time': total execution time,
            'model': trained knn model,
            'predictions': test predictions,
            'y_test': true test labels
        }
    """
    print("=" * 60)
    print("MNIST Classification - K-Nearest Neighbors (KNN)")
    print("=" * 60)

    total_start = time.time()
    
    # initialize visualization
    viz = Visualization()

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

    # pre-compute distances once (this is the expensive operation)
    print("\nstep 3: pre-computing distances from all test samples to all training samples...")
    print("(this will be reused for all k values)")
    compute_start = time.time()
    all_distances = compute_all_distances(x_train, x_test)
    compute_time = time.time() - compute_start
    print(f"distance computation took {compute_time:.2f} seconds")

    # try different k values using pre-computed distances
    k_values = [1, 3, 5, 7, 9]  # note, bare minimum is 1, 3, 5
    results = {}
    timings = {}
    all_predictions = {}

    print("\nstep 4: testing different k values with pre-computed distances...")
    print("=" * 60)

    for k in k_values:
        print(f"\n--- testing knn with k={k} ---")
        k_start = time.time()

        # make predictions using pre-computed distances
        print(f"making predictions on {len(x_test)} test images...")
        predict_start = time.time()
        predictions = predict_with_k(all_distances, y_train, k)
        predict_time = time.time() - predict_start

        accuracy = calculate_accuracy(predictions, y_test)

        # save for stats reporting
        k_total_time = time.time() - k_start
        results[k] = accuracy
        timings[k] = {"predict": predict_time, "total": k_total_time}
        all_predictions[k] = predictions

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
        print(f"  predict time: {timings[k]['predict']:.2f}s")
        print(f"  total time: {timings[k]['total']:.2f}s")

    # more stats, namely which k did best
    best_k = max(results, key=results.get)
    best_accuracy = results[best_k]
    best_predictions = all_predictions[best_k]
    
    print(
        f"\nbest k value: {best_k} with accuracy {best_accuracy:.4f} ({best_accuracy*100:.2f}%)"
    )

    print(f"\ndistance computation time: {compute_time:.2f} seconds")
    print(
        f"total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    
    # create visualizations for best model
    print("\nstep 5: generating visualizations...")
    viz.plot_confusion_matrix(y_test, best_predictions, f"KNN_k{best_k}")
    
    print("\n" + "=" * 60)
    print("knn training and testing complete!")
    print("=" * 60)
    
    # return results for the best k value
    # note: train_time is set to compute_time since that's the main "training" overhead
    return {
        'accuracy': best_accuracy * 100,  # convert to percentage
        'train_time': compute_time,  # distance computation is the "training" phase
        'total_time': total_time,
        'model': KNN(k=best_k),  # create new model with best k (would need to refit if used later)
        'predictions': best_predictions,
        'y_test': y_test,
        'best_k': best_k,  # extra info specific to knn
        'all_k_results': results  # keep track of all k values tested
    }


def main():
    """standalone execution - just calls train()"""
    train()


if __name__ == "__main__":
    main()