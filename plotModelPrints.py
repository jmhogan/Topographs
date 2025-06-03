import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("Outputs/training_log.csv")  # Change to your actual filename

# Map mode from True/False to Train/Val for clarity
df['mode'] = df['mode'].map({True: 'Train', False: 'Val', 'True': 'Train', 'False': 'Val'})

# List of metrics
loss_cols = [col for col in df.columns if col.startswith('loss')]
init_cols = [col for col in df.columns if col.startswith('Initialisation')]
reg_cols = [col for col in df.columns if col.startswith('Regression')]
class_cols = [col for col in df.columns if col.startswith('Classification')]

# Set general matplotlib style for "pretty" plots
plt.style.use('ggplot')
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

def plot_metrics(df, metrics, title, ylabel, filename):
    plt.figure(figsize=(8, 6))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    markers = ['o', 's', '^', 'D', 'v', '*']
    for i, mode in enumerate(df['mode'].unique()):
        subdf = df[df['mode'] == mode]
        for j, metric in enumerate(metrics):
            plt.plot(
                subdf['epoch'],
                subdf[metric],
                marker=markers[j % len(markers)],
                color=colors[j % len(colors)],
                linestyle='-' if mode == 'Train' else '--',
                label=f"{metric} ({mode})"
            )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

# Plot Losses
plot_metrics(df, loss_cols, "Loss by Epoch", "Loss", "./Outputs/loss_plot.png")

# Plot Initialisation
plot_metrics(df, init_cols, "Initialisation by Epoch", "Initialisation", "./Outputs/init_plot.png")

# Plot Regressions
plot_metrics(df, reg_cols, "Regression Losses by Epoch", "Regression Loss", "./Outputs/regression_plot.png")

# Plot Classifications
plot_metrics(df, class_cols, "Classification Losses by Epoch", "Classification Loss", "./Outputs/classification_plot.png")
