import numpy as np


class NaiveBayes:
    """
    naive bayes classifier using numpy only.
    uses binarized pixels and assumes pixel independence to calculate probabilities.
    """

    def __init__(self):
        """
        initialize naive bayes classifier.
        """
        self.priors = None  # # probability of each digit appearing in the dataset (10,)
        self.likelihoods = None  # stores the probability of a pixel being on for each digit. will be a 10 by 784 array between 0, 1
        self.n_classes = 10  # digits 0-9

    def fit(self, x_train, y_train):
        """
        learn probability distributions from training data.

        args:
            x_train: training images, shape (n_samples, 784), values in [0, 1]
            y_train: training labels, shape (n_samples,)
        """
        n_samples, n_features = x_train.shape

        # binarize the training data (1 if pixel > 0.5, else 0)
        x_train_binary = (x_train > 0.5).astype(int)

        # initialize storage for probabilities
        self.priors = np.zeros(self.n_classes) # list that's 10 elemenets long
        self.likelihoods = np.zeros((self.n_classes, n_features)) # 10 rows, 784 columns

        # for each digit, calculate probabilities
        for digit in range(self.n_classes):
            # get all images of this digit
            digit_images = x_train_binary[y_train == digit]

            # calculate prior (relative to the entire dataset)
            self.priors[digit] = len(digit_images) / n_samples # this counts rows

            # calculate likelihood for each pixel (relative to other digits of the same type)
            # add smoothing to avoid zero probabilities
            self.likelihoods[digit] = (digit_images.sum(axis=0) + 1) / (
                len(digit_images) + 2 # this counts by column (we're comparing pixels to pixels)
            )

        print(
            f"naive bayes fitted with {n_samples} training samples across {self.n_classes} classes"
        )

    def predict(self, x_test):
        """
        predict labels for test data using bayes' theorem.

        args:
            x_test: test images, shape (n_samples, 784), values in [0, 1]

        returns:
            predictions: predicted labels, shape (n_samples,)
        """
        # binarize test data
        x_test_binary = (x_test > 0.5).astype(int)

        predictions = []

        # predict each test sample
        for i, test_sample in enumerate(x_test_binary):
            # show progress every 1000 samples (faster than knn so less frequent updates)
            if (i + 1) % 1000 == 0:
                print(f"predicting sample {i + 1}/{len(x_test)}...")

            pred = self._predict_single(test_sample)
            predictions.append(pred)

        return np.array(predictions)

    def _predict_single(self, test_sample):
        """
        predict the label for a single test sample using bayes' theorem.

        args:
            test_sample: single binarized image, shape (784,)

        returns:
            predicted label (0-9)
        """
        # calculate posterior probability for each digit
        posteriors = []
        for digit in range(self.n_classes):
            # start with prior: p(digit)
            # use log probabilities to avoid numerical underflow
            log_posterior = np.log(self.priors[digit])

            # multiply by likelihood for each pixel
            # for pixels that are "on" (1), use p(pixel=1 | digit)
            # for pixels that are "off" (0), use p(pixel=0 | digit) = 1 - p(pixel=1 | digit)
            for pixel_idx in range(len(test_sample)):
                if test_sample[pixel_idx] == 1:
                    log_posterior += np.log(self.likelihoods[digit][pixel_idx])
                else:
                    log_posterior += np.log(1 - self.likelihoods[digit][pixel_idx])

            posteriors.append(log_posterior)

        # return digit with highest posterior probability
        return np.argmax(posteriors)