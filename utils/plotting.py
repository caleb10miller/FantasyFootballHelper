import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
import time
def plot_actual_vs_predicted(actual, predicted, title="Actual vs Predicted Values", 
                           xlabel="Actual Values", ylabel="Predicted Values",
                           figsize=(10, 8), save_path=None):
    """
    Create a scatter plot of actual vs predicted values with a perfect prediction line.
    
    Parameters:
    -----------
    actual : array-like
        The actual target values
    predicted : array-like
        The predicted values from the model
    title : str, optional
        Title for the plot
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path
    """
    # Calculate R² score
    r2 = r2_score(actual, predicted)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    sns.scatterplot(x=actual, y=predicted, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Customize the plot
    plt.title(f"{title}\nR² = {r2:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)