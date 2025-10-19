import numpy as np
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent)) # this piece of code I need so my training file can easily find where src is

# self-made files
from src.utils.data_loader import load, split
from src.models.cnn import CNN
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
    train and evaluate convolutional neural network.
    
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
    print("MNIST Classification - Convolutional Neural Network (CNN)")
    print("=" * 60)

    total_start = time.time()
    
    # initialize visualization
    viz = Visualization()

    # set device
    device = torch.device("cpu")
    print(f"using device: {device}")

    # load images from disk (keep 2d for cnn!)
    print("\nstep 1: loading mnist data...")
    load_start = time.time()
    images, labels = load(DATA_DIR="MNIST", FLATTEN=False, NORMALIZE=True)  # keep 2d!
    load_time = time.time() - load_start
    print(f"data loading took {load_time:.2f} seconds")

    # shuffle and split into train/test
    print("\nstep 2: splitting into train/test sets...")
    split_start = time.time()
    x_train, x_test, y_train, y_test = split(images, labels, TEST_SIZE=0.2, SEED=42)
    split_time = time.time() - split_start
    print(f"data splitting took {split_time:.2f} seconds")

    # convert to pytorch tensors and reshape for cnn
    print("\nstep 3: converting to pytorch tensors...")
    # reshape to (batch, channels, height, width) format
    x_train_tensor = torch.FloatTensor(x_train).unsqueeze(1)  # (48000, 28, 28) -> (48000, 1, 28, 28)
    y_train_tensor = torch.LongTensor(y_train)
    x_test_tensor = torch.FloatTensor(x_test).unsqueeze(1)    # (12000, 28, 28) -> (12000, 1, 28, 28)
    y_test_tensor = torch.LongTensor(y_test)

    # create datasets and dataloaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"created dataloaders with batch size {batch_size}")
    print(f"image shape for cnn: {x_train_tensor[0].shape}")

    # initialize model, loss function, and optimizer
    print("\nstep 4: initializing model...")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print(f"model architecture: conv(1->32, 3x3) -> relu -> maxpool -> conv(32->64, 3x3) -> relu -> maxpool -> fc -> output")
    print(f"model: {model}")
    print(f"loss function: cross-entropy (includes softmax)")
    print(f"optimizer: sgd with learning rate 0.01 and momentum 0.9")

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

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

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
    viz.plot_confusion_matrix(y_test, predictions, "CNN")
    viz.plot_training_history(history['losses'], history['test_accuracies'], "CNN")

    print("\n" + "=" * 60)
    print("cnn training and testing complete!")
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