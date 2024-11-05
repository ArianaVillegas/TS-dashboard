import streamlit as st
import pandas as pd
from aeon.benchmarking import get_estimator_results_as_array, get_available_estimators
from aeon.datasets.tsc_datasets import univariate_equal_length


# Set up Streamlit page configuration
st.set_page_config(
    page_title="TS Dashboard",
    page_icon="📈",
    layout="wide"
)

# Cache loading of classifiers
@st.cache_data
def load_classifiers():
    # return list(uni_classifiers_2023.keys())
    cls = get_available_estimators(task="classification", return_dataframe=False)
    return cls

# Load accuracy results and cache for efficiency
@st.cache_data
def load_all_accuracy_results():
    # accuracy_results = get_bake_off_2023_results(default_only=False)
    datasets = list(univariate_equal_length)
    classifiers = load_classifiers()
    default_split_some, names = get_estimator_results_as_array(
        estimators=classifiers, datasets=datasets
    )
    results_df = pd.DataFrame(
        default_split_some, 
        columns=classifiers, 
        index=datasets
    )
    return results_df

# Load your custom results from CSV and calculate average accuracy
@st.cache_data
def load_custom_results(csv_path):
    custom_df = pd.read_csv(csv_path)
    custom_df["method"] = custom_df["method"] + " - " + custom_df["mode"]
    custom_df = custom_df.groupby(["method", "dataset"]).accuracy.mean().unstack()
    return custom_df.T

# Calculate average accuracy once
@st.cache_data
def get_top_methods(num_methods=15):
    all_accuracy_df = load_all_accuracy_results()
    average_accuracy = all_accuracy_df.mean()
    return average_accuracy.nlargest(num_methods).index.tolist()

# Helper function to highlight top results
def style_best_results(df):
    def highlight_row(row):
        row = (row * 100).round(2)  # Convert to percentage
        sorted_row = row.sort_values(ascending=False)
        row = row.astype(str)
        
        # Apply bold and underline styles
        row[sorted_row.index[0]] = f'<strong style="color:yellow;">{row[sorted_row.index[0]]}</strong>'
        if len(row) > 1:
            row[sorted_row.index[1]] = f'<strong style="color:green;">{row[sorted_row.index[1]]}</strong>'
        return row
    
    return df.apply(highlight_row, axis=1)

# Main UI
st.title("📈 TS Dashboard")
st.header("Accuracy Comparison of Time Series Classifiers")
st.write("This dashboard compares accuracy results for various time series classifiers across multiple datasets.")

# Initialize data and UI elements
classifiers = load_classifiers()
print(classifiers)
top_methods = get_top_methods(10)
selected_methods = st.multiselect("Select Methods to Display:", classifiers, default=top_methods)

# File uploader for custom results CSV
# csv_path = st.file_uploader("Upload CSV with custom method results", type=["csv"])
csv_path = "classification_results.csv"
custom_results = None
if csv_path:
    custom_results = load_custom_results(csv_path)

# Load and filter accuracy results dynamically based on selection
if selected_methods:
    accuracy_df = load_all_accuracy_results()[selected_methods]
    
    # If custom results are uploaded, merge with benchmark results
    if custom_results is not None:
        # Align index/columns for merging
        common_datasets = accuracy_df.index.intersection(custom_results.index)
        accuracy_df = accuracy_df.loc[common_datasets]
        print(custom_results)
        print(common_datasets)
        print(custom_results.loc[common_datasets])
        custom_results = custom_results.loc[common_datasets]
        
        # Concatenate benchmark and custom results side-by-side
        combined_df = pd.concat([accuracy_df, custom_results], axis=1)
    else:
        combined_df = accuracy_df
    
    # Sort and style data
    combined_df['max_accuracy'] = combined_df.max(axis=1)  # Add max accuracy
    combined_df = combined_df.sort_values(by='max_accuracy').drop(columns=['max_accuracy'])
    styled_df = style_best_results(combined_df)
    styled_html = styled_df.to_html(escape=False, index=True)
    
    # Custom CSS for table styling
    st.markdown(
        """
        <style>
        .styled-table {
            margin: auto;
            border-collapse: collapse;
            overflow: hidden;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
        }
        .styled-table th, .styled-table td {
            padding: 15px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .styled-table tr:hover {
            background-color: #686868;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f'<div class="styled-table">{styled_html}</div>', unsafe_allow_html=True)
else:
    st.write("Please select at least one method to display results.")
