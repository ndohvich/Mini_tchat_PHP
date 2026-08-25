"""Publication-oriented figures created only when their inputs exist."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def _save(fig,name):
    Path("results/figures").mkdir(parents=True,exist_ok=True);fig.savefig(f"results/figures/{name}.png",dpi=300,bbox_inches="tight");fig.savefig(f"results/figures/{name}.pdf",bbox_inches="tight");plt.close(fig)
def create_core_figures(df,datetime_column,target,pearson,performance):
    """Create factual target evolution, distribution, correlation and performance figures."""
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(df[datetime_column],df[target],lw=.7,label=target);ax.set(title="Temporal evolution of observed radiation",xlabel="Time",ylabel=target);ax.legend();_save(fig,"figure_1_temporal_evolution")
    fig,ax=plt.subplots(figsize=(6,4));ax.hist(df[target].dropna(),bins=40);ax.set(title="Distribution of target variable",xlabel=target,ylabel="Count");_save(fig,"figure_2_target_distribution")
    if not pearson.empty:
        fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(pearson, vmin=-1,vmax=1,cmap="coolwarm");ax.set_xticks(range(len(pearson)),pearson.columns,rotation=90);ax.set_yticks(range(len(pearson)),pearson.index);fig.colorbar(im,ax=ax,label="Pearson correlation");ax.set_title("Correlation analysis (descriptive, non-causal)");_save(fig,"figure_3_correlation")
    if not performance.empty:
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(performance.model,performance.RMSE);ax.tick_params(axis="x",rotation=40);ax.set(title="Locked-test model comparison",ylabel="RMSE");_save(fig,"figure_10_model_comparison")
