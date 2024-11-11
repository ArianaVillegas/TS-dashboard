import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "classification_results.csv"

def load_custom_results(csv_path):
    custom_df = pd.read_csv(csv_path)
    custom_df["method"] = custom_df["method"] + " - " + custom_df["mode"]
    custom_df = custom_df.groupby(["method", "dataset"]).accuracy.mean().unstack()
    return custom_df.T
custom_results = load_custom_results(csv_path)

print(custom_results.head())

plt.scatter(custom_results["NN - fine-tune"], custom_results["NN - pretrained"])
plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))
plt.show()