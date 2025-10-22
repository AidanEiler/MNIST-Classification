import sys
import time
from pathlib import Path

# add parent directory to path so we can run from anywhere
sys.path.append(str(Path(__file__).parent.parent))

# use relative imports since we're inside src
from training.train_knn import train as train_knn
from training.train_naive_bayes import train as train_naive_bayes
from training.train_linear import train as train_linear
from training.train_mlp import train as train_mlp
from training.train_cnn import train as train_cnn
from utils.visualization import Visualization


def main():
    print("=" * 80)
    print("MNIST CLASSIFICATION - RUNNING ALL EXPERIMENTS")
    print("=" * 80)
    print("\nThis will train all 5 models and generate visualizations.")
    print("Estimated total time: ~3-5 minutes\n")
    
    overall_start = time.time()
    
    # initialize visualization utility
    viz = Visualization()
    
    # dictionary to store all results
    all_results = {}
    
    # ========== 1. K-NEAREST NEIGHBORS ==========
    print("\n" + "=" * 80)
    print("1/5 - TRAINING K-NEAREST NEIGHBORS")
    print("=" * 80)
    knn_results = train_knn()
    all_results['KNN (k=3)'] = knn_results
    
    # ========== 2. NAIVE BAYES ==========
    print("\n" + "=" * 80)
    print("2/5 - TRAINING NAIVE BAYES")
    print("=" * 80)
    nb_results = train_naive_bayes()
    all_results['Naive Bayes'] = nb_results
    
    # ========== 3. LINEAR CLASSIFIER ==========
    print("\n" + "=" * 80)
    print("3/5 - TRAINING LINEAR CLASSIFIER")
    print("=" * 80)
    linear_results = train_linear()
    all_results['Linear'] = linear_results
    
    # ========== 4. MULTILAYER PERCEPTRON ==========
    print("\n" + "=" * 80)
    print("4/5 - TRAINING MULTILAYER PERCEPTRON")
    print("=" * 80)
    mlp_results = train_mlp()
    all_results['MLP'] = mlp_results
    
    # ========== 5. CONVOLUTIONAL NEURAL NETWORK ==========
    print("\n" + "=" * 80)
    print("5/5 - TRAINING CONVOLUTIONAL NEURAL NETWORK")
    print("=" * 80)
    cnn_results = train_cnn()
    all_results['CNN'] = cnn_results
    
    # ========== GENERATE COMPARISON VISUALIZATIONS ==========
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    # comparison bar chart
    viz.plot_comparison_bar(all_results, title="MNIST Model Accuracy Comparison")
    
    # comparison table
    viz.generate_comparison_table(all_results)
    
    # ========== SUMMARY ==========
    overall_time = time.time() - overall_start
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print(f"total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    
    # find best model
    best_model = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
    best_accuracy = all_results[best_model]['accuracy']
    
    print(f"\nbest model: {best_model} with {best_accuracy:.2f}% accuracy")
    
    print("\nvisualizations saved to: out/visualization/")
    print("- Confusion matrices for each model")
    print("- Training history plots (Linear, MLP, CNN)")
    print("- Weight visualizations (Linear)")
    print("- Probability maps (Naive Bayes)")
    print("- Model comparison chart")
    print("- Model comparison table")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()