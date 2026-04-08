import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sensitivity analysis results.")
    parser.add_argument(
        '-n', '--nnunet_csv',
        type=str,
        required=True,
        help="Path to the CSV file containing sensitivity analysis results for the nnunet model.",
    )
    parser.add_argument(
        '-c', '--cellpose_csv',
        type=str,
        required=True,
        help="Path to the CSV file containing sensitivity analysis results for the cellpose model.",
    )
    args = parser.parse_args()

    nnunet_df = pd.read_csv(args.nnunet_csv)
    cellpose_df = pd.read_csv(args.cellpose_csv)

    plt.figure(figsize=(10, 8))
    sns.set_context("notebook", font_scale=2)  # Increase font size
    sns.set_style('ticks')
    sns.set_palette('mako_r', n_colors=2)
    plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust tick label font size
    sns.lineplot(data=nnunet_df, x='threshold', y='f1', label='nnU-Net')
    sns.lineplot(data=cellpose_df, x='threshold', y='f1', label='Cellpose', linestyle='dashed')
    plt.title('Sensitivity analysis of IoU threshold on F1 Score')
    plt.xlabel('IoU Threshold', labelpad=5)
    plt.ylabel('F1 Score')
    plt.xlim((0.3, 0.99))
    plt.ylim((0, 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('f1_sensitivity_analysis_plot.png', dpi=300)
    print("Sensitivity analysis plot saved as 'f1_sensitivity_analysis_plot.png'")

    # now, we do the same for precision and recall
    plt.figure(figsize=(10, 8))
    sns.set_context("notebook", font_scale=2)  # Increase font size
    sns.set_style('ticks')
    sns.set_palette('mako_r', n_colors=2)
    plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust tick label font size
    sns.lineplot(data=nnunet_df, x='threshold', y='precision', label='nnU-Net')
    sns.lineplot(data=cellpose_df, x='threshold', y='precision', label='Cellpose', linestyle='dashed')
    plt.title('Sensitivity analysis of IoU threshold on Precision')
    plt.xlabel('IoU Threshold', labelpad=5)
    plt.ylabel('Precision')
    plt.xlim((0.3, 0.99))
    plt.ylim((0, 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('precision_sensitivity_analysis_plot.png', dpi=300)
    print("Sensitivity analysis plot saved as 'precision_sensitivity_analysis_plot.png'")

    plt.figure(figsize=(10, 8))
    sns.set_context("notebook", font_scale=2)
    sns.set_style('ticks')
    sns.set_palette('mako_r', n_colors=2)
    plt.tick_params(axis='both', which='major', labelsize=20)
    sns.lineplot(data=nnunet_df, x='threshold', y='recall', label='nnU-Net')
    sns.lineplot(data=cellpose_df, x='threshold', y='recall', label='Cellpose', linestyle='dashed')
    plt.title('Sensitivity analysis of IoU threshold on Recall')
    plt.xlabel('IoU Threshold', labelpad=5)
    plt.ylabel('Recall')
    plt.xlim((0.3, 0.99))
    plt.ylim((0, 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('recall_sensitivity_analysis_plot.png', dpi=300)
    print("Sensitivity analysis plot saved as 'recall_sensitivity_analysis_plot.png'")