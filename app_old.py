import streamlit as st
import pandas as pd
from aeon.benchmarking import get_bake_off_2023_results, uni_classifiers_2023
from aeon.datasets.tsc_datasets import univariate_equal_length

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="TS Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load available classifiers
@st.cache_data
def load_classifiers():
    return list(uni_classifiers_2023.keys())

classifiers = load_classifiers()
all_methods = classifiers
# all_methods = classifiers ["classification"].unique()

# Load datasets and accuracy results dynamically
@st.cache_data
def load_accuracy_results(selected_methods):
    accuracy_results = get_bake_off_2023_results(default_only=False)
    results_df = pd.DataFrame(accuracy_results, columns=all_methods, index=list(univariate_equal_length))
    results_df = results_df[selected_methods]
    # results_df = pd.DataFrame(accuracy_results, columns=selected_methods, index=list(univariate_equal_length))
    # results_df.index.name = "dataset_name"
    
    # Sort datasets by difficulty (ascending max accuracy)
    results_df['max_accuracy'] = results_df.max(axis=1)  # Calculate max accuracy per row
    results_df = results_df.sort_values(by='max_accuracy')  # Sort by max accuracy
    results_df.drop(columns=['max_accuracy'], inplace=True)  # Remove helper column after sorting
    
    return results_df

# Helper function to highlight top results
def style_best_results(df):
    def highlight_row(row):
        row = (row * 100).round(2)
        sorted_row = row.sort_values(ascending=False)
        row = row.astype(str)
        
        # Apply bold and underline styles
        row[sorted_row.index[0]] = f'<strong style="color:yellow;">{row[sorted_row.index[0]]}</strong>'
        if len(row) > 1:
            row[sorted_row.index[1]] = f'<strong style="color:green;">{row[sorted_row.index[1]]}</strong>'
        return row
    
    return df.apply(highlight_row, axis=1)

# UI for selecting classifiers
st.title("📈 TS Dashboard")
st.header("Accuracy Comparison of Time Series Classifiers")
st.write("This dashboard compares accuracy results for various time series classifiers across multiple datasets.")

# Load all accuracy results to find the top 5 methods
all_accuracy_df = load_accuracy_results(all_methods)

# Calculate the average accuracy for each method and select top 5
average_accuracy = all_accuracy_df.mean()
top_methods = average_accuracy.nlargest(15).index.tolist()

# Multiselect for methods (top 5 selected by default)
selected_methods = st.multiselect("Select Methods to Display:", all_methods, default=top_methods)

# Load and filter data based on selected methods
if selected_methods:
    accuracy_df = load_accuracy_results(selected_methods)
    
    # Style the results and convert to HTML with custom CSS
    styled_df = style_best_results(accuracy_df)
    styled_html = styled_df.to_html(escape=False, index=True)
    
    # Custom CSS for table styling and layout
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
