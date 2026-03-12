"""
Master Experiment Runner

Runs all key experiments for the EFTA paper in sequence.

Usage:
    python experiments/run_all_experiments.py
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.validate_leakage import main as validate_data
from datetime import datetime

def print_header(text):
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def print_section(text):
    print("\n" + "-"*80)
    print(text)
    print("-"*80)

def main():
    start_time = time.time()
    
    print_header("EFTA EXPERIMENT SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 0: Validate data pipeline
    print_section("STEP 0: Data Leakage Validation")
    print("Running validation checks...")
    validation_passed = validate_data()
    
    if not validation_passed:
        print("\n[ERROR] Data validation failed! Fix issues before proceeding.")
        return 1
    
    print("\n[PASS] Data pipeline validated successfully!")
    
    # Step 1: Main EFTA training (quick test with single seed)
    print_section("STEP 1: EFTA Training on MNIST")
    print("Running: python experiments/train_efta.py")
    print("(This will train EFTA configurations on MNIST)")
    print("\nNote: For statistical significance, run separately:")
    print("  python experiments/advanced/01_statistical_significance.py")
    
    # Step 2: Fashion-MNIST
    print_section("STEP 2: Fashion-MNIST Evaluation")
    print("Running: python experiments/fashion_mnist_efta.py")
    print("(Tests generalization to more complex dataset)")
    
    # Step 3: Branch Visualization
    print_section("STEP 3: Branch Specialization Analysis")
    print("Running: python experiments/advanced/03_branch_visualization.py")
    print("(Visualizes what different leaves learn)")
    
    # Summary
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print_header("EXPERIMENT SUMMARY")
    print(f"Total runtime: {hours}h {minutes}m")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutputs saved to:")
    print("  - outputs/plots/  (visualizations)")
    print("  - outputs/results/ (numerical results)")
    print("\nNext steps:")
    print("  1. Review results in outputs/ directory")
    print("  2. Run statistical significance: python experiments/advanced/01_statistical_significance.py")
    print("  3. Generate paper figures from outputs/plots/")
    
    return 0


if __name__ == '__main__':
    print("\n" + "="*80)
    print("EFTA PAPER EXPERIMENTS".center(80))
    print("="*80)
    print("\nThis script will run:")
    print("  1. Data validation (leakage checks)")
    print("  2. EFTA training on MNIST")
    print("  3. Fashion-MNIST evaluation")
    print("  4. Branch visualization")
    print("\nEstimated total runtime: 3-4 hours")
    print("\nNote: You can run each step individually:")
    print("  - python experiments/validate_leakage.py")
    print("  - python experiments/train_efta.py")
    print("  - python experiments/fashion_mnist_efta.py")
    print("  - python experiments/advanced/03_branch_visualization.py")
    print("\n" + "="*80)
    
    response = input("\nProceed with all experiments? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled. Run individual scripts as needed.")
        sys.exit(0)
    
    exit(main())
