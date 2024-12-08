from pathlib import Path
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.arrow import ArrowWriter

from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.datasets import load_classification


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    y: Union[List[np.int16], np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
    )

    # Set an arbitrary start time
    start = np.datetime64("2000-01-01 00:00", "s")

    dataset = [
        {"start": start, "target": ts, "class": ys} for ts, ys in zip(time_series, y)
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


if __name__ == "__main__":
    folder = 'datasets/random'
    dataset = 'PigAirwayPressure'
    print(f'dataset {dataset}')
    
    split = 'train' # ['train', 'test']
    
    print(f'dataset {dataset} {split}')
    X, y = load_classification(name=dataset, split=split)  
    X = X.squeeze()
    print(X.shape, y.shape)
    print(np.unique(y))
    
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(4, 6, figsize=(16, 12))  # 2 rows, 3 columns

    # Loop through each time series and plot it
    for i, ax in enumerate(axes.flatten()):
        if i < len(X):  # Ensure there's data to plot
            ax.plot(X[2*i])
            ax.set_title(f"Class: {y[2*i]}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
        else:  # For unused subplots, hide the axes
            ax.axis("off")

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
    
    # Convert to GluonTS arrow format
    # convert_to_arrow(f"datasets/{dataset}_{split}.arrow", time_series=X, y=y)