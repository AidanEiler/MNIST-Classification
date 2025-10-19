import numpy as np
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

# self-made files
from src.utils.data_loader import load, split
from src.models.linear import LinearClassifier
from src.utils.visualization import Visualization


def calculate_accuracy(model, data_loader, device):
    """
    calculate accuracy on a dataset.

    args:
        model: trained model
        data_loader: dataloader with test data
        device: cpu or cuda

    returns:
        accuracy as float
    """
    model.eval()  # set to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # don't calculate gradients for evaluation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # get index of max score

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def get_predictions(model, data_loader, device):
    """
    get all predictions from model on a dataset.
    
    args:
        model: trained model
        data_loader: dataloader
        device: cpu or cuda
        
    returns:
        predictions as numpy array
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    return np.array(all_predictions)


def train():
    """
    train and evaluate linear classifier.
    
    returns:
        dict: {
            'accuracy': final test accuracy as percentage,
            'train_time': time spent training,
            'total_time': total execution time,
            'model': trained model,
            'predictions': test predictions,
            'y_test': true test labels,
            'history': {
                'losses': list of losses per epoch,
                'train_accuracies': list of train accuracies per epoch,
                'test_accuracies': list of test accuracies per epoch
            }
        }
    """
    print("=" * 60)
    print("MNIST Classification - Linear Classifier")
    print("=" * 60)

    total_start = time.time()
    
    # initialize visualization
    viz = Visualization()

    # set device - I tried using CUDA for this, but my GPU is so new that pytorch doesn't support it
    device = torch.device("cpu")
    print(f"using device: {device}")

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

    # convert to pytorch tensors
    print("\nstep 3: converting to pytorch tensors...")
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.LongTensor(y_test)

    # create datasets and dataloaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"created dataloaders with batch size {batch_size}")

    # initialize model, loss function, and optimizer
    print("\nstep 4: initializing model...")
    model = LinearClassifier().to(device)
    criterion = nn.MSELoss()  # L2 loss
    """
    criterion = (
        nn.CrossEntropyLoss()
    )  # better for classification than mse (make note in report)
    """
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9
    )  # stochastic gradient descent, higher lr + momentum (make note in report)

    print(f"model: {model}")
    print(f"loss function: mse (l2 loss)")
    print(f"optimizer: sgd with learning rate 0.1 and momentum 0.9")

    # training loop
    print("\nstep 5: training...")
    print("=" * 60)

    num_epochs = 20
    train_start = time.time()
    
    # track history for visualization
    history = {
        'losses': [],
        'train_accuracies': [],
        'test_accuracies': []
    }

    for epoch in range(num_epochs):
        model.train()  # set to training mode
        epoch_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels_one_hot = torch.zeros(labels.size(0), 10).to(device) # comment out if using cross entropy
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1) # comment out if using cross entropy

            # forward pass (the forward() method is invoked here for this)
            outputs = model(inputs)
            loss = criterion(outputs, labels_one_hot) # would use labels directly if using cross entropy

            # backward pass and optimization
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()  # calculate gradients
            optimizer.step()  # update weights

            epoch_loss += loss.item()

        # calculate average loss for this epoch
        avg_loss = epoch_loss / len(train_loader)
        
        # always track metrics every epoch for accurate plotting
        train_acc = calculate_accuracy(model, train_loader, device)
        test_acc = calculate_accuracy(model, test_loader, device)
        history['losses'].append(avg_loss)
        history['train_accuracies'].append(train_acc * 100)
        history['test_accuracies'].append(test_acc * 100)

        # print progress every few epochs to avoid clutter
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"epoch [{epoch+1}/{num_epochs}], loss: {avg_loss:.4f}, "
                f"train acc: {train_acc:.4f} ({train_acc*100:.2f}%), "
                f"test acc: {test_acc:.4f} ({test_acc*100:.2f}%)"
            )
        else:
            print(f"epoch [{epoch+1}/{num_epochs}], loss: {avg_loss:.4f}")

    train_time = time.time() - train_start

    # final evaluation
    print("\nstep 6: final evaluation...")
    final_test_acc = calculate_accuracy(model, test_loader, device)
    predictions = get_predictions(model, test_loader, device)

    total_time = time.time() - total_start

    # show final results
    print("\n" + "=" * 60)
    print("summary of results:")
    print("=" * 60)
    print(f"training time: {train_time:.2f} seconds")
    print(f"total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"final test accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    
    # create visualizations
    print("\nstep 7: generating visualizations...")
    viz.plot_confusion_matrix(y_test, predictions, "Linear")
    viz.plot_training_history(history['losses'], history['test_accuracies'], "Linear")
    viz.visualize_linear_weights(model, "Linear")

    print("\n" + "=" * 60)
    print("linear classifier training and testing complete!")
    print("=" * 60)
    
    # return standardized results
    return {
        'accuracy': final_test_acc * 100,  # convert to percentage
        'train_time': train_time,
        'total_time': total_time,
        'model': model,
        'predictions': predictions,
        'y_test': y_test,
        'history': history
    }


def main():
    """standalone execution - just calls train()"""
    train()


if __name__ == "__main__":
    main()