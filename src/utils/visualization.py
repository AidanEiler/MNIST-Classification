import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix


class Visualization:
    """
    visualization utilities for mnist classification project.
    handles confusion matrices, training curves, weight visualizations, and comparisons.
    """

    def __init__(self, output_dir="out/visualization"):
        """
        initialize visualization utility.

        args:
            output_dir: directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # set consistent style
        plt.rcParams['figure.figsize'] = (10, 8)

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        create and save confusion matrix heatmap.

        args:
            y_true: true labels
            y_pred: predicted labels
            model_name: name for saving the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # create heatmap using imshow
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        # add colorbar
        plt.colorbar(im, ax=ax)
        
        # set ticks and labels
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels(range(10))
        ax.set_yticklabels(range(10))
        
        # add text annotations
        for i in range(10):
            for j in range(10):
                text = ax.text(j, i, str(cm[i, j]),
                             ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
        
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        
        output_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"saved confusion matrix to {output_path}")

    def plot_training_history(self, losses, accuracies, model_name):
        """
        plot training loss and accuracy curves over epochs.

        args:
            losses: list of loss values per epoch
            accuracies: list of accuracy values per epoch
            model_name: name for saving the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = list(range(1, len(losses) + 1))
        
        # plot loss
        ax1.plot(epochs, losses, "b-", linewidth=2)
        ax1.set_title(f"{model_name} - Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        # force integer x-axis ticks starting from 1
        if len(epochs) <= 20:
            ax1.set_xticks(epochs)
        else:
            # if too many epochs, show every 5th
            ax1.set_xticks(range(1, len(epochs) + 1, 5))
        
        # plot accuracy
        ax2.plot(epochs, accuracies, "g-", linewidth=2)
        ax2.set_title(f"{model_name} - Test Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, alpha=0.3)
        # force integer x-axis ticks starting from 1
        if len(epochs) <= 20:
            ax2.set_xticks(epochs)
        else:
            # if too many epochs, show every 5th
            ax2.set_xticks(range(1, len(epochs) + 1, 5))
        
        output_path = self.output_dir / f"{model_name}_training_history.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"saved training history to {output_path}")

    def visualize_linear_weights(self, model, model_name="linear"):
        """
        visualize linear classifier weight matrix as digit images.

        args:
            model: trained linear classifier model
            model_name: name for saving the plot
        """
        # get weights from model (shape: 10, 784)
        weights = model.linear.weight.data.cpu().numpy()
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle("Linear Classifier Learned Weights (What Each Digit Looks Like)", fontsize=14)
        
        for i, ax in enumerate(axes.flat):
            # reshape to 28x28 image
            weight_image = weights[i].reshape(28, 28)
            
            # plot with diverging colormap (blue=negative, red=positive)
            im = ax.imshow(weight_image, cmap="RdBu", aspect="auto")
            ax.set_title(f"Digit {i}")
            ax.axis("off")
        
        # add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        
        output_path = self.output_dir / f"{model_name}_weights.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"saved weight visualization to {output_path}")

    def visualize_naive_bayes_probs(self, model, model_name="naive_bayes"):
        """
        visualize naive bayes probability maps as digit images.

        args:
            model: trained naive bayes model
            model_name: name for saving the plot
        """
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle("Naive Bayes Probability Maps (Pixel 'On' Probabilities)", fontsize=14)
        
        for i, ax in enumerate(axes.flat):
            # reshape likelihoods to 28x28 image
            prob_image = model.likelihoods[i].reshape(28, 28)
            
            # plot with hot colormap (dark=low prob, bright=high prob)
            im = ax.imshow(prob_image, cmap="hot", aspect="auto", vmin=0, vmax=1)
            ax.set_title(f"Digit {i}")
            ax.axis("off")
        
        # add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="P(pixel=1|digit)")
        
        output_path = self.output_dir / f"{model_name}_probabilities.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"saved probability visualization to {output_path}")

    def plot_comparison_bar(self, results_dict, title="Model Accuracy Comparison"):
        """
        create bar chart comparing model accuracies.

        args:
            results_dict: dictionary of {model_name: {'accuracy': value, ...}}
            title: plot title
        """
        models = list(results_dict.keys())
        accuracies = [results_dict[m]['accuracy'] for m in models]
        
        # convert to percentages if needed
        if max(accuracies) <= 1.0:
            accuracies = [acc * 100 for acc in accuracies]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies, color='steelblue', edgecolor='black')
        
        # add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{acc:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.ylim(0, 105)  # give room for labels
        plt.grid(axis="y", alpha=0.3)
        
        output_path = self.output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"saved comparison chart to {output_path}")

    def generate_comparison_table(self, results_dict, save_to_file=True):
        """
        generate and display comparison table of all models.

        args:
            results_dict: dictionary of {model_name: {'accuracy': ..., 'train_time': ..., ...}}
            save_to_file: whether to save table to text file
        
        returns:
            dict with formatted table data
        """
        # print header
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Model':<20} {'Accuracy (%)':<15} {'Train Time (s)':<18} {'Total Time (s)':<15}")
        print("-" * 80)
        
        # prepare data for file
        table_lines = []
        table_lines.append("=" * 80)
        table_lines.append("MODEL COMPARISON TABLE")
        table_lines.append("=" * 80)
        table_lines.append(f"{'Model':<20} {'Accuracy (%)':<15} {'Train Time (s)':<18} {'Total Time (s)':<15}")
        table_lines.append("-" * 80)
        
        # print and collect rows
        for model_name, results in results_dict.items():
            accuracy = f"{results['accuracy']:.2f}"
            train_time = f"{results.get('train_time', 0):.2f}"
            total_time = f"{results.get('total_time', 0):.2f}"
            
            row = f"{model_name:<20} {accuracy:<15} {train_time:<18} {total_time:<15}"
            print(row)
            table_lines.append(row)
        
        print("=" * 80 + "\n")
        table_lines.append("=" * 80)
        
        # save to file
        if save_to_file:
            output_path = self.output_dir / "model_comparison_table.txt"
            with open(output_path, 'w') as f:
                f.write('\n'.join(table_lines))
            
            print(f"saved comparison table to {output_path}")
        
        return results_dict