import numpy as np
from PIL import Image
from pathlib import Path


def load(DATA_DIR="MNIST", FLATTEN=True, NORMALIZE=True):
    """
    load mnist images from the dataset folder

    args:
        DATA_DIR: path to mnist folder containing 0-9 subfolders
        FLATTEN: if TRUE, flatten images to 784-dim vectors. if false, keep as 28x28. I'll likely set this to FALSE when training the CNN
        NORMALIZE: if true, normalize pixel values to [0, 1]

    returns:
        images: numpy array (n_samples, 784) if flattened, or (n_samples, 28, 28) if not flattened
        labels: numpy array (n_samples) containing digit labels 0-9
    """
    images = []
    labels = []

    # loop through each digit folder (0-9)
    for digit in range(10):
        folder_path = Path(DATA_DIR) / str(digit)

        # skip if this digit's folder doesn't exist
        if not folder_path.exists():
            print(f"warning: folder {folder_path} does not exist")
            continue

        # grab all the png files in this folder
        image_files = list(folder_path.glob("*.png"))
        print(f"loading {len(image_files)} images for digit {digit}...")

        for img_path in image_files:
            img = Image.open(img_path).convert(
                "L"
            )  # L here is luminance, which is apparently greyscale mode
            # we shouldn't need to do this since our images are already black and white,
            # but this is to be careful

            # convert to a numpy array
            img_array = np.array(img, dtype=np.float32)

            if NORMALIZE:
                img_array = img_array / 255.0

            if FLATTEN:
                img_array = img_array.flatten()

            # add this image and its label to our lists
            images.append(img_array)
            labels.append(digit)

    # convert python lists to numpy arrays for efficiency
    images = np.array(images)
    labels = np.array(labels)

    print(f"\ntotal images loaded: {len(images)}")
    print(f"image shape: {images.shape}")
    print(f"labels shape: {labels.shape}")

    return images, labels


def split(IMAGES, LABELS, TEST_SIZE=0.2, SEED=42):
    """
    split data into training and testing sets with shuffling.

    args:
        IMAGES: numpy array of images
        LABELS: numpy array of labels
        TEST_SIZE: fraction of data to use for testing (default 0.2 = 20%)
        SEED: random seed for reproducibility

    returns:
        x_train: training images (n_train, 784) or (n_train, 28, 28)
        x_test: test images (n_test, 784) or (n_test, 28, 28)
        y_train: training labels (n_train,) with digit values 0-9
        y_test: test labels (n_test,) with digit values 0-9
    """
    # set random seed so we get the same split every time
    np.random.seed(SEED)

    # figure out how many images we have total
    n_samples = len(IMAGES)

    # create an array of indices and shuffle them randomly
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # figure out where to split (80% train, 20% test by default)
    split_idx = int(n_samples * (1 - TEST_SIZE))

    # divide the shuffled indices into train and test groups
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # use the indices to split our actual data
    x_train = IMAGES[train_indices]
    x_test = IMAGES[test_indices]
    y_train = LABELS[train_indices]
    y_test = LABELS[test_indices]

    print(f"\ntrain set: {len(x_train)} images")
    print(f"test set: {len(x_test)} images")
    print(f"train/test split: {(1-TEST_SIZE)*100:.0f}/{TEST_SIZE*100:.0f}")

    return x_train, x_test, y_train, y_test
