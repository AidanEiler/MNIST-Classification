import numpy as np


class KNN:
    """
    k-nearest neighbors classifier using numpy only.
    stores training data and classifies by finding the k closest neighbors.
    """

    def __init__(self, k=3):
        """
        args:
            k: number of nearest neighbors to use for voting
        """
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        """
        store the training data. note: knn doesn't actually train anything just memorizes.

        args:
            x_train: training images, shape (n_samples, 784)
            y_train: training labels, shape (n_samples,)
        """
        self.x_train = x_train
        self.y_train = y_train
        print(f"knn (k={self.k}) fitted with {len(x_train)} training samples")

    def predict(self, x_test):
        """
        predict labels for test data by finding k nearest neighbors.

        args:
            x_test: test images, shape (n_samples, 784)

        returns:
            predictions: predicted labels, shape (n_samples,)
        """
        predictions = []

        # predict each test sample one at a time
        for i, test_sample in enumerate(x_test):
            # show progress every 100 samples
            if (i + 1) % 100 == 0:
                print(f"predicting sample {i + 1}/{len(x_test)}...")

            # find the k nearest neighbors and get majority vote
            pred = self._predict_single(test_sample)
            predictions.append(pred)

        return np.array(predictions)

    def _predict_single(
        self, test_sample
    ):  # private method, shouldn't be called outside of class
        """
        predict the label for a single test sample.

        args:
            test_sample: single image, shape (784,)

        returns:
            predicted label (0-9)
        """
        # compute euclidean distance from this test sample to all training samples
        distances = self._compute_distances(test_sample)

        k_nearest_indices = np.argsort(distances)[
            : self.k
        ]  # returns the INDICIES that would sort the array, not the elements themselves this is
        # an important distinction because we want to know the images that the small distances
        # map to

        # get the labels of these k nearest neighbors
        k_nearest_labels = self.y_train[k_nearest_indices]

        # return the most common label (majority vote)
        # np.bincount counts occurrences of each label, argmax finds the most common
        most_common = np.bincount(k_nearest_labels).argmax()

        return most_common

    def _compute_distances(
        self, test_sample
    ):  # private method, shouldn't be called outside of class
        """
        compute euclidean distance from test sample to all training samples.

        args:
            test_sample: single image, shape (784,)

        returns:
            distances: array of distances to each training sample, shape (n_train_samples,)
        """
        # euclidean distance: sqrt(sum((x1 - x2)^2))
        # we can vectorize this for all training samples at once (aka no need to refer to by-element operations)
        distances = np.sqrt(np.sum((self.x_train - test_sample) ** 2, axis=1))

        return distances
