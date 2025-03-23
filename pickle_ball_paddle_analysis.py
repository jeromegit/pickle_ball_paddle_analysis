import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and layout
st.set_page_config(
    page_title="Pickleball Paddle Analyzer",
    layout="wide"
)

# Title and introduction
st.title("Pickleball Paddle Data Analysis")
st.write("Analyze and explore relationships between different pickleball paddle characteristics.")


def massage_data(data_df):
    massaged_data_df = data_df.dropna(subset=['Paddle', 'Price'])

    columns_to_drop = ['Discount Code/Link', 'Discount', 'Discounted Price']
    massaged_data_df = massaged_data_df.drop(columns=columns_to_drop)

    massaged_data_df['Price'] = massaged_data_df['Price'].str.replace('$', '', regex=False).astype(float).round(0).astype(int)

    # Remove commas from numeric columns and convert to numeric
    massaged_data_df = massaged_data_df.replace({',': ''}, regex=True)

    # Convert percentage columns to float in the range 0 to 1
    percent_cols = [col for col in massaged_data_df.columns if col.endswith('%')]
    for col in percent_cols:
        massaged_data_df[col] = massaged_data_df[col].str.replace('%', '').astype(float) / 100

    return massaged_data_df


# Function to load and process data
def load_data():
    FILE_PATH = '/Users/jerome/projects/pickle_ball_paddle_analysis/john_knew_pickleball_paddle_database.tsv'
    try:
        data_df = pd.read_csv(FILE_PATH, sep='\t')
        massaged_data_df = massage_data(data_df)
    except Exception as e:
        st.error(f"Error loading data from file:{FILE_PATH} {e}")
        st.stop()


    return massaged_data_df


# Load data based on user choice
df = load_data()

# Data overview
st.header("Data Overview")
st.write(f"Dataset has {df.shape[0]} paddles and {df.shape[1]} features.")

# Display data
with st.expander("View Raw Data"):
    st.dataframe(df)

# Display summary statistics
with st.expander("Summary Statistics"):
    st.write(df.describe())

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Sidebar for feature selection
st.sidebar.header("Analysis Options")

# Correlation analysis
st.header("Correlation Analysis")
corr_features = st.multiselect(
    "Select features for correlation analysis:",
    numeric_cols,
    default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
)

if len(corr_features) > 1:
    corr = df[corr_features].corr()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax)
    plt.title('Correlation Matrix')
    st.pyplot(fig)

    # Feature relationship exploration
    st.header("Feature Relationship Explorer")
    x_axis = st.selectbox("X-axis", corr_features)
    y_axis = st.selectbox("Y-axis", corr_features, index=1 if len(corr_features) > 1 else 0)
    color_by = st.selectbox("Color by", ["None"] + categorical_cols)

    fig, ax = plt.subplots(figsize=(10, 6))
    if color_by != "None":
        scatter = sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=color_by, ax=ax)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        scatter = sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)

    plt.title(f"{y_axis} vs {x_axis}")
    plt.tight_layout()
    st.pyplot(fig)

    # Display correlation coefficient
    corr_value = df[x_axis].corr(df[y_axis])
    st.write(f"Correlation coefficient between {x_axis} and {y_axis}: **{corr_value:.3f}**")
else:
    st.write("Please select at least two features for correlation analysis")

# Distribution analysis
st.header("Distribution Analysis")
dist_feature = st.selectbox("Select feature to view distribution:", numeric_cols)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df[dist_feature], kde=True, ax=ax)
plt.title(f"Distribution of {dist_feature}")
st.pyplot(fig)

# Box plots by categories
st.header("Category Comparison")
if categorical_cols:
    cat_feature = st.selectbox("Select categorical feature:", categorical_cols)
    num_feature = st.selectbox("Select numerical feature to compare:", numeric_cols,
                               index=1 if len(numeric_cols) > 1 else 0)

    # Filter for categories with sufficient data
    cat_counts = df[cat_feature].value_counts()
    valid_cats = cat_counts[cat_counts >= 1].index.tolist()

    if len(valid_cats) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=cat_feature, y=num_feature, data=df[df[cat_feature].isin(valid_cats)], ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{num_feature} by {cat_feature}")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("Not enough categories with sufficient data for comparison")
else:
    st.write("No categorical features found in the dataset")

# Feature explorer
st.header("Feature Explorer")
st.write("Compare multiple paddle features side by side")

feature_explorer_cols = st.multiselect(
    "Select features to compare:",
    df.columns.tolist(),
    default=df.columns[:4].tolist() if len(df.columns) > 4 else df.columns.tolist()
)

if feature_explorer_cols:
    # Allow users to filter by category
    if categorical_cols:
        filter_cat = st.selectbox("Filter by category:", ["None"] + categorical_cols)
        if filter_cat != "None":
            filter_value = st.selectbox("Select value:", df[filter_cat].unique())
            filtered_df = df[df[filter_cat] == filter_value]
        else:
            filtered_df = df
    else:
        filtered_df = df

    st.write(filtered_df[feature_explorer_cols])

# Paddle comparison
st.header("Paddle Comparison")
if "Paddle" in df.columns:
    paddles = st.multiselect(
        "Select paddles to compare:",
        df["Paddle"].unique(),
        default=df["Paddle"].unique()[:2].tolist() if len(df["Paddle"].unique()) > 2 else df["Paddle"].unique().tolist()
    )

    if paddles:
        comparison_df = df[df["Paddle"].isin(paddles)]
        comparison_features = st.multiselect(
            "Select features for comparison:",
            numeric_cols,
            default=["Price", "Swing Weight", "Spin RPM", "Serve Speed-MPH (Power)"] if all(f in numeric_cols for f in
                                                                                            ["Price", "Swing Weight",
                                                                                             "Spin RPM",
                                                                                             "Serve Speed-MPH (Power)"]) else numeric_cols[
                                                                                                                              :4]
        )

        if comparison_features:
            # Radar chart for comparing paddles
            paddle_values = []
            for paddle in paddles:
                values = comparison_df[comparison_df["Paddle"] == paddle][comparison_features].values[0]
                paddle_values.append(values)

            # Normalize values for radar chart
            min_vals = comparison_df[comparison_features].min()
            max_vals = comparison_df[comparison_features].max()
            normalized_values = []

            for values in paddle_values:
                norm_values = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                               for v, min_val, max_val in zip(values, min_vals, max_vals)]
                normalized_values.append(norm_values)

            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(comparison_features), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Close the loop

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

            for i, paddle in enumerate(paddles):
                values = normalized_values[i]
                values = np.concatenate((values, [values[0]]))  # Close the loop
                ax.plot(angles, values, 'o-', linewidth=2, label=paddle)
                ax.fill(angles, values, alpha=0.1)

            ax.set_thetagrids(angles[:-1] * 180 / np.pi, comparison_features)
            plt.title('Paddle Comparison')
            plt.legend(loc='upper right')
            st.pyplot(fig)
