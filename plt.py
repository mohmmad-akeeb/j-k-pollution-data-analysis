import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =========================
# 1. LOAD & PREPARE DATA
# =========================
with open("results.json", "r") as f:
    results = json.load(f)

# Convert to a Pandas DataFrame
data = []
for model, metrics in results.items():
    data.append({
        "Model": model,
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"]
    })
df = pd.DataFrame(data)

# =========================
# 2. SET PUBLICATION STYLE
# =========================
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

# =========================
# 3. CREATE OUTPUT FOLDER
# =========================
output_dir = "Final Plots"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 4. PLOTTING FUNCTION
# =========================
def save_journal_plot(df, metric, title, ylabel, filename):
    plt.figure(figsize=(7, 5))
    
    # Create the barplot
    ax = sns.barplot(
        data=df, 
        x="Model", 
        y=metric, 
        hue="Model", 
        palette="viridis", 
        dodge=False, 
        legend=False
    )
    
    # ADDED: Annotate the bars with the exact values
    # fmt='%.3f' limits the text to 3 decimal places. Change to '%.2f' for 2 decimal places.
    # padding=3 gives a little space between the top of the bar and the number.
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontweight='bold', fontsize=10)
    
    # Typography and labels
    plt.title(title, fontweight='bold', pad=15)
    plt.ylabel(ylabel, fontweight='bold')
    plt.xlabel("Model", fontweight='bold')
    
    # Rotate x-ticks and align right
    plt.xticks(rotation=45, ha='right')
    
    # Remove top and right borders 
    sns.despine()
    
    # Save with tight bounding box to prevent cut-off labels
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# =========================
# 5. GENERATE PLOTS
# =========================
save_journal_plot(df, "RMSE", "RMSE Comparison", "RMSE", "rmse_comparison.png")
save_journal_plot(df, "MAE", "MAE Comparison", "MAE", "mae_comparison.png")
save_journal_plot(df, "R2", "R² Score Comparison", "R² Score", "r2_comparison.png")

print(f"Publication-ready plots saved in: {output_dir}")