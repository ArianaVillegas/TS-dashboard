import os
import matplotlib
import streamlit as st
import pandas as pd
import altair as alt
from pymongo import MongoClient
from dotenv import load_dotenv
from matplotlib.patches import Rectangle
from aeon.benchmarking.results_loaders import get_available_estimators, get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.visualisation import (
    plot_boxplot,
    plot_critical_difference,
    plot_pairwise_scatter,
)

matplotlib.rcParams.update({'font.size': 12})

# Set up Streamlit page configuration
st.set_page_config(
    page_title="TS Dashboard",
    page_icon="📈",
    layout="wide"
)

# MongoDB connection setup
# Load environment variables from .env file
load_dotenv()

# Retrieve the MongoDB URI
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
@st.cache_resource
def get_mongo_client():
    if not MONGO_URI:
        raise ValueError("MongoDB URI is not set in the .env file.")
    return MongoClient(MONGO_URI)

client = get_mongo_client()
db = client["ts-dashboard"]
collection_results = db["classification"]
collection_meta = db["meta_cls"]

# Check the active Streamlit theme (light or dark)
theme = st.get_option("theme.base")
if theme == "dark":
    background_color = "black"
    text_color = "white"
    line_color = "cyan"
    grid_color = "gray"
else:
    background_color = "white"
    text_color = "black"
    line_color = "blue"
    grid_color = "lightgray"

# Cache classifiers
@st.cache_data
def load_classifiers():
    cls = get_available_estimators(task="classification")["classification"]
    return list(cls)

# Fetch data from MongoDB
@st.cache_data
def fetch_from_mongodb(collection_name, filter_query=None, projection=None):
    collection = db[collection_name]
    cursor = collection.find(filter_query or {}, projection)
    return pd.DataFrame(list(cursor))

# Load accuracy results and cache
@st.cache_data
def load_all_accuracy_results():
    classifiers = load_classifiers()
    datasets = list(univariate_equal_length)
    default_split_some, _ = get_estimator_results_as_array(
        estimators=classifiers, datasets=datasets, measure="balacc"
    )
    return pd.DataFrame(
        default_split_some, 
        columns=classifiers, 
        index=datasets
    )

# Load custom results from MongoDB
@st.cache_data
def load_custom_results():
    # Fetch classification results from MongoDB
    custom_df = fetch_from_mongodb("classification")
    custom_df["method"] = custom_df["method"] + " - " + custom_df["mode"]
    custom_df = custom_df.groupby(["method", "dataset"]).balacc.mean().unstack()
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
        row[sorted_row.index[0]] = f'<strong style="color:green;">{row[sorted_row.index[0]]}</strong>'
        if len(row) > 1:
            row[sorted_row.index[1]] = f'<strong style="color:#CC6600;">{row[sorted_row.index[1]]}</strong>'
        return row
    
    return df.apply(highlight_row, axis=1)

# Main UI
st.title("📈 TS Dashboard")
st.header("Balanced Accuracy Comparison of Time Series Classifiers")
st.write("This dashboard compares balanced accuracy results for various time series classifiers across multiple datasets.")

# Custom results
custom_results = load_custom_results()
custom_results = custom_results.loc[list(univariate_equal_length)]
select_methods = ["NN - pretrained-small", "NN - pretrained-large", "CNN - pretrained-small", "NN - fine-tune-1"]

# Initialize data and UI elements
classifiers = load_classifiers() + list(custom_results.columns)
paper_methods = ['HC2', 'MR-Hydra', 'RDST', 'H-InceptionTime', 'WEASEL-2.0', 'QUANT', 'FreshPRINCE', 'PF']
top_methods = get_top_methods(6)
top_methods = list(set(top_methods)) + select_methods
selected_methods = st.multiselect("Select Methods to Display:", classifiers, default=top_methods)

if selected_methods:
    accuracy_df = load_all_accuracy_results()
    
    # If custom results are loaded, merge with benchmark results
    if custom_results is not None:
        meta_df = fetch_from_mongodb("meta-cls")
        meta_df.set_index("dataset", inplace=True)
        combined_df = pd.concat([meta_df, accuracy_df, custom_results], axis=1)
        
        # Add tooltip information to each dataset name
        combined_df.index = combined_df.apply(
            lambda row: f'<span class="tooltip">{row.name} '
                        f'<span class="tooltiptext">Classes: {int(row["classes"])}<br>'
                        f'Length: {int(row["length"])}<br>'
                        f'Channels: {int(row["channels"])}<br>'
                        f'Train: {int(row["train"])}<br>'
                        f'Test: {int(row["test"])}</span></span>',
            axis=1
        )
        combined_df = combined_df.drop(columns=["classes", "length", "channels", "train", "test"])   
    else:
        combined_df = accuracy_df
    
    combined_df = combined_df[selected_methods]
    combined_df["max_accuracy"] = combined_df.max(axis=1)
    combined_df = combined_df.sort_values(by="max_accuracy").drop(columns=["max_accuracy"])
        
    styled_df = style_best_results(combined_df)
    styled_html = styled_df.to_html(escape=False, index=True)
    
    # Custom CSS for tooltip styling
    st.markdown(
        """
        <style>
        .tooltip {
            position: relative;
            cursor: pointer;
            color: blue;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 140px;
            background-color: black;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 10; /* Ensure it's above the table */
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            margin-left: -70px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 0.85;
        }
        .styled-table {
            margin: auto;
            border-collapse: collapse;
            overflow: visible; /* Allow content to overflow out of table bounds */
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 5%;
        }
        .styled-table th, .styled-table td {
            padding: 15px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .styled-table tr:hover {
            background-color: #C7C7C7;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="styled-table">{styled_html}</div>', unsafe_allow_html=True)
    
    def set_background(ax, background_color):
        ax.set_facecolor(background_color)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=background_color, zorder=-1))
    
    # Display the two plots side by side in Streamlit
    st.title("🧐 Methods Comparison")
    col1, col2 = st.columns([11, 10])
    methods = list(combined_df.columns)

    with col1:
        # Critical Difference Plot
        st.subheader("Critical difference plot")
        fig_cd, ax_cd = plot_critical_difference(
            combined_df.to_numpy(),
            methods,
            test="wilcoxon",
            correction="holm",
        )
        st.pyplot(fig_cd)
        
    with col1:
        # Violin Plot
        st.subheader("Swarm plot")
        fig_violin, ax_violin = plot_boxplot(
            combined_df.to_numpy(),
            methods,
            plot_type="swarm",
        )
        st.pyplot(fig_violin)

    with col2:
        # Pairwise Scatter Plot
        st.subheader("Pairwise scatter plot")
        col_x, col_y = st.columns(2)
        with col_y:
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 0px; /* Remove margin space */
                }
                </style>
                <div class="custom-label">X-axis method:</div>
                """, 
                unsafe_allow_html=True
            )
            axis_y = st.selectbox("", methods, index=methods.index("NN - pretrained-large"), label_visibility="collapsed")
        with col_x:
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 0px; /* Remove margin space */
                }
                </style>
                <div class="custom-label">Y-axis method:</div>
                """, 
                unsafe_allow_html=True
            )
            axis_x = st.selectbox("", methods, index=methods.index("NN - pretrained-small"), label_visibility="collapsed")
        fig_ps, ax_ps = plot_pairwise_scatter(
            combined_df[axis_x], combined_df[axis_y], axis_x, axis_y
        )
        st.pyplot(fig_ps)
        
    
    # Display the two plots side by side in Streamlit
    st.title("🎯 Model bias")
    best_results = accuracy_df.max(axis=1)
    ts_length, num_classes = st.columns(2)
    
    with ts_length:
        st.subheader("TimeSeries Length")
        method = st.selectbox("", list(custom_results.columns), index=select_methods.index("NN - pretrained-small"), label_visibility="collapsed", key="tsl")
        sel_method = custom_results[[method]]
        sel_method["dataset"] = sel_method.index
        sel_method["diff"] = sel_method[method] - best_results
        sel_method["length"] = meta_df["length"]
        print("SEL METHOD")
        print(sel_method)
        scatter = alt.Chart(sel_method).mark_circle(size=100).encode(
            x=alt.X("length", title="Time Series Length"),
            y=alt.Y("diff", title="Accuracy Difference"),
            tooltip=["dataset", "length", "diff"]
        ).properties(
            width=700,
            height=500
        )
        st.altair_chart(scatter, use_container_width=True)
    
    with num_classes:
        st.subheader("Number of Classes")
        method = st.selectbox("", list(custom_results.columns), index=select_methods.index("NN - pretrained-small"), label_visibility="collapsed", key="nc")
        sel_method = custom_results[[method]]
        sel_method["dataset"] = sel_method.index
        sel_method["diff"] = sel_method[method] - best_results
        sel_method["classes"] = meta_df["classes"]
        print("SEL METHOD")
        print(sel_method)
        scatter = alt.Chart(sel_method).mark_circle(size=100).encode(
            x=alt.X("classes", title="# Classes"),
            y=alt.Y("diff", title="Accuracy Difference"),
            tooltip=["dataset", "classes", "diff"]
        ).properties(
            width=700,
            height=500
        )
        st.altair_chart(scatter, use_container_width=True)
        
    length_classes, placeholder = st.columns(2)
    with length_classes:
        st.subheader("TimeSeries Length vs Number of Classes")
        method = st.selectbox("", list(custom_results.columns), index=select_methods.index("NN - pretrained-small"), label_visibility="collapsed", key="tsvsc")
        sel_method = custom_results[[method]]
        sel_method["dataset"] = sel_method.index
        sel_method["diff"] = sel_method[method] - best_results
        sel_method["length"] = meta_df["length"]
        sel_method["classes"] = meta_df["classes"]
        print("SEL METHOD")
        # Define the custom color scale
        min_diff = sel_method["diff"].min()
        max_diff = sel_method["diff"].max()
        custom_scale = alt.Scale(
            domain=[min_diff, 0, max_diff],  # Min value, midpoint (0), max value
            range=["red", "lightgray", "green"]  # Negative -> red, neutral -> gray, positive -> green
        )
        scatter = alt.Chart(sel_method).mark_circle(size=100).encode(
            x=alt.X("length", title="Time Series Length"),
            y=alt.Y("classes", title="Number of Classes"),
            color=alt.Color("diff", scale=custom_scale, title="Accuracy Difference"),
            tooltip=["dataset", "length", "classes", "diff"]
        ).properties(
            title="Time Series Length vs Number of Classes",
            width=700,
            height=500
        )
        st.altair_chart(scatter, use_container_width=True)
else:
    st.write("Please select at least one method to display results.")
