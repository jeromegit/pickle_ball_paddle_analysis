import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


class PaddleAnalysis:
    """Class for analyzing pickleball paddle data"""
    
    # Page name constants
    PAGE_DATA_OVERVIEW = "Data Overview"
    PAGE_PRICE_ANALYSIS = "Price Analysis"
    PAGE_CORRELATION = "Correlation Analysis"
    PAGE_DISTRIBUTION = "Feature Distribution"
    PAGE_COMPARISON = "Paddle Comparison"
    PAGE_CUSTOM = "Custom Analysis"
    
    def __init__(self, file_path="john_knew_pickleball_paddle_database.tsv"):
        """Initialize the paddle analysis app"""
        self.df = self.load_data(file_path)
        
        # Define column types
        self.numeric_cols = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
        self.categorical_cols = [col for col in self.df.columns if col not in self.numeric_cols and col != 'Paddle']
        
        # Calculate percentile columns if they don't exist
        self.calculate_percentiles()

    def load_data(self, file_path):
        FILE_PATH = file_path
        try:
            data_df = pd.read_csv(FILE_PATH, sep='\t')
            massaged_data_df = self.massage_data(data_df)

            return massaged_data_df
        except Exception as e:
            st.error(f"Error loading data from file:{FILE_PATH} {e}")
            st.stop()

        return None

    def massage_data(self, data_df):
        massaged_data_df = data_df.dropna(subset=['Paddle', 'Price', 'Spin RPM'])

        columns_to_drop = ['Discount Code/Link', 'Discount', 'Discounted Price']
        massaged_data_df = massaged_data_df.drop(columns=columns_to_drop)

        # Clean up price/spin and make the numeric
        massaged_data_df['Price'] = massaged_data_df['Price'].str.replace('$', '', regex=False).astype(float).round(
            0).astype(int)
        massaged_data_df['Spin RPM'] = massaged_data_df['Spin RPM'].str.replace(',', '', regex=True).astype(
            float).round(0).astype(int)

        # Convert percentage columns to float
        percent_cols = [col for col in massaged_data_df.columns if col.endswith('%')]
        for col in percent_cols:
            massaged_data_df[col] = massaged_data_df[col].str.replace('%', '').astype(float)

        return massaged_data_df

    def calculate_percentiles(self):
        """Calculate percentile columns for key metrics"""
        # Define the metrics to calculate percentiles for
        percentile_metrics = {
            'Spin RPM': 'Spin Percentile',
            'Swing Weight': 'Swing Weight Percentile',
            'Twist Weight': 'Twist Weight Percentile',
            'Balance Point (cm)': 'Balance Point Percentile',
            'Firepower (0-100)': 'Power Percentile',
            'Punch Volley Speed-MPH (Pop)': 'Pop Percentile'
        }
        
        # Calculate percentiles for each metric
        for metric, percentile_col in percentile_metrics.items():
            if metric in self.df.columns:
                # For some metrics, higher is better, for others lower is better
                if metric in ['Balance Point (cm)', 'Twist Weight']:
                    # For these metrics, lower is better, so reverse the percentile
                    self.df[percentile_col] = (1 - self.df[metric].rank(pct=True)) * 100
                else:
                    # For these metrics, higher is better
                    self.df[percentile_col] = self.df[metric].rank(pct=True) * 100
                
                # Add the new column to numeric_cols if not already there
                if percentile_col not in self.numeric_cols:
                    self.numeric_cols.append(percentile_col)
        
        # Make sure Price is in numeric_cols if it exists
        if 'Price' in self.df.columns and 'Price' not in self.numeric_cols:
            self.numeric_cols.append('Price')

    def plot_correlation_matrix(self, features=None):
        """Plot correlation matrix for selected numeric features"""
        if features is None:
            # Use default numeric columns if none specified
            corr_features = self.df[self.numeric_cols]
        else:
            corr_features = self.df[features]

        # Calculate correlation
        corr = corr_features.corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(4, 3))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set color map
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Plot heatmap
        heatmap = sns.heatmap(corr, mask=mask, cmap=cmap,
                              annot=True, square=True, linewidths=.5,
                              fmt=".2f", center=0, ax=ax, annot_kws={"size": 5})

        # Reduce font sizes for axis labels and ticks
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=5, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
        ax.set_title("Correlation Matrix", fontsize=7)

        # Reduce colorbar font size
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=4)

        plt.tight_layout()
        return fig

    def plot_scatter(self, x_axis, y_axis, color_by=None):
        """Plot scatter plot of two features with optional coloring"""
        fig, ax = plt.subplots(figsize=(3.5, 2))

        if color_by and color_by in self.df.columns:
            if color_by in self.numeric_cols:
                # For numeric columns, use a colormap
                scatter = ax.scatter(
                    self.df[x_axis],
                    self.df[y_axis],
                    c=self.df[color_by],
                    alpha=0.7,
                    s=20,
                    cmap='viridis'
                )
                # Add a color bar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_by, fontsize=7)
                cbar.ax.tick_params(labelsize=6)
            else:
                # For categorical columns, use different colors for each category
                categories = self.df[color_by].unique()
                for category in categories:
                    subset = self.df[self.df[color_by] == category]
                    ax.scatter(
                        subset[x_axis],
                        subset[y_axis],
                        label=category,
                        alpha=0.7,
                        s=20
                    )
                # Add legend for categorical coloring
                ax.legend(fontsize=6, title=color_by, title_fontsize=7, 
                         loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3)
        else:
            ax.scatter(self.df[x_axis], self.df[y_axis], alpha=0.7, s=20)

        ax.set_xlabel(x_axis, fontsize=8)
        ax.set_ylabel(y_axis, fontsize=8)
        ax.set_title(f"{x_axis} vs {y_axis}", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=7)

        # Add correlation text
        corr_value = self.df[x_axis].corr(self.df[y_axis])
        ax.text(0.05, 0.95, f"Correlation: {corr_value:.2f}",
                transform=ax.transAxes, fontsize=7, color='blue')

        plt.tight_layout()
        return fig

    def plot_distribution(self, dist_feature):
        """Plot distribution of a numeric feature"""
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.histplot(self.df[dist_feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {dist_feature}", fontsize=8)
        ax.set_xlabel(dist_feature, fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=6)
        plt.tight_layout()
        return fig

    def plot_categorical_analysis(self, cat_feature, num_feature):
        """Plot relationship between categorical and numeric features"""
        # Count values in each category
        cat_counts = self.df[cat_feature].value_counts()
        valid_cats = cat_counts[cat_counts >= 3].index.tolist()

        fig, ax = plt.subplots(figsize=(5, 3))

        # Create appropriate visualization (e.g., boxplot)
        filtered_data = self.df[self.df[cat_feature].isin(valid_cats)]
        sns.boxplot(x=cat_feature, y=num_feature, data=filtered_data, ax=ax)

        ax.set_title(f"{num_feature} by {cat_feature}", fontsize=8)
        ax.set_xlabel(cat_feature, fontsize=7)
        ax.set_ylabel(num_feature, fontsize=7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
        ax.tick_params(axis='y', which='major', labelsize=6)
        plt.tight_layout()

        return fig

    def filter_data(self, filter_cat=None, filter_value=None):
        """Filter the dataframe based on category and value"""
        filtered_df = self.df.copy()

        if filter_cat and filter_cat in self.df.columns:
            if filter_value:
                if isinstance(filter_value, list):
                    filtered_df = filtered_df[filtered_df[filter_cat].isin(filter_value)]
                else:
                    filtered_df = filtered_df[filtered_df[filter_cat] == filter_value]

        return filtered_df

    def compare_paddles(self, paddles, comparison_features=None):
        """Compare selected paddles across features"""
        if comparison_features is None:
            comparison_features = self.numeric_cols

        comparison_df = self.df[self.df['Paddle'].isin(paddles)].copy()

        # Create visualization for comparison
        # This could be a bar chart, radar chart, etc.

        return comparison_df

    def create_radar_chart(self, paddle_values, feature_order=None, title=None):
        """Create radar chart for comparing paddles with specified feature order"""
        # If no feature order is provided, use a default order
        if feature_order is None:
            feature_order = [
                'Spin RPM',
                'Twist Weight',
                'Balance Point (cm)',
                'Swing Weight',
                'Punch Volley Speed-MPH (Pop)',
                'Firepower (0-100)'
            ]
        
        # Filter feature_order to only include features that exist in the data
        available_features = list(list(paddle_values.values())[0].keys())
        ordered_features = [f for f in feature_order if f in available_features]
        
        # Add any remaining features that weren't in our predefined order
        for feature in available_features:
            if feature not in ordered_features:
                ordered_features.append(feature)
        
        # Normalize values for radar chart
        min_vals = {}
        max_vals = {}
        normalized_values = {}

        # Calculate min and max for each feature
        for paddle, values in paddle_values.items():
            for feature, value in values.items():
                if feature not in min_vals or value < min_vals[feature]:
                    min_vals[feature] = value
                if feature not in max_vals or value > max_vals[feature]:
                    max_vals[feature] = value

        # Normalize values between 0 and 1
        for paddle, values in paddle_values.items():
            norm_values = {}
            for feature, value in values.items():
                min_val = min_vals[feature]
                max_val = max_vals[feature]
                if max_val == min_val:
                    norm_values[feature] = 0.5
                else:
                    norm_values[feature] = (value - min_val) / (max_val - min_val)
            normalized_values[paddle] = norm_values

        # Create radar chart using ordered features
        angles = np.linspace(0, 2 * np.pi, len(ordered_features), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, (paddle, values) in enumerate(normalized_values.items()):
            # Use the ordered features to get values in the right order
            ordered_values = [values[feature] for feature in ordered_features]
            # Add the first value at the end to close the loop
            ordered_values = np.concatenate((ordered_values, [ordered_values[0]]))
            ax.plot(angles, ordered_values, linewidth=1.5, label=paddle)
            ax.fill(angles, ordered_values, alpha=0.1)

        # Add feature labels around the chart in the specified order
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(ordered_features, fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=12, pad=20)

        # Add legend with smaller font and place it outside the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)

        plt.tight_layout()
        return fig

    def run_app(self):
        """Main function to run the Streamlit app"""
        # Set page title and layout
        st.set_page_config(
            page_title="Pickleball Paddle Analyzer",
            layout="wide"
        )

        st.title("Pickleball Paddle Analysis")

        # Sidebar for navigation
        page = st.sidebar.selectbox(
            "Choose Analysis...",
            [
                self.PAGE_DATA_OVERVIEW,
                self.PAGE_PRICE_ANALYSIS,
                self.PAGE_CORRELATION,
                self.PAGE_DISTRIBUTION,
                self.PAGE_COMPARISON,
                self.PAGE_CUSTOM
            ]
        )

        if page == self.PAGE_DATA_OVERVIEW:
            self.show_data_overview()
            
        elif page == self.PAGE_PRICE_ANALYSIS:
            self.show_price_analysis()

        elif page == self.PAGE_CORRELATION:
            self.show_correlation_analysis()

        elif page == self.PAGE_DISTRIBUTION:
            self.show_feature_distribution()

        elif page == self.PAGE_COMPARISON:
            self.show_paddle_comparison()

        elif page == self.PAGE_CUSTOM:
            self.show_custom_analysis()

    def show_data_overview(self):
        """Display data overview page"""
        st.header(self.PAGE_DATA_OVERVIEW)
        
        # General dataset statistics
        st.subheader("Dataset Statistics")
        
        # Create columns for key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Paddles", len(self.df))
            
        with col2:
            st.metric("Companies", self.df['Company'].nunique())
            
        with col3:
            # Calculate average price
            avg_price = self.df['Price'].mean()
            st.metric("Average Price", f"${int(round(avg_price))}")
            
        with col4:
            # Calculate price range
            min_price = self.df['Price'].min()
            max_price = self.df['Price'].max()
            st.metric("Price Range", f"${int(round(min_price))} - ${int(round(max_price))}")
        
        # Display categorical column distributions
        st.subheader("Categorical Distributions")
        
        # Create tabs for each categorical column with reasonable number of categories
        categorical_cols = [col for col in self.categorical_cols 
                           if col != 'Condition' and self.df[col].nunique() <= 15 and self.df[col].nunique() > 1]
        
        if categorical_cols:
            tabs = st.tabs(categorical_cols)
            
            for i, col in enumerate(categorical_cols):
                with tabs[i]:
                    # Get value counts and calculate percentages
                    counts = self.df[col].value_counts().reset_index()
                    counts.columns = [col, 'Count']
                    counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(1)
                    counts['Percentage'] = counts['Percentage'].astype(str) + '%'
                    
                    # Add price statistics for each category
                    price_stats = self.df.groupby(col)['Price'].agg(['min', 'max', 'mean']).reset_index()
                    
                    # Format price columns - round to nearest integer
                    price_stats['Min Price'] = price_stats['min'].apply(lambda x: f"${int(round(x))}")
                    price_stats['Max Price'] = price_stats['max'].apply(lambda x: f"${int(round(x))}")
                    price_stats['Avg Price'] = price_stats['mean'].apply(lambda x: f"${int(round(x))}")
                    
                    # Merge counts with price stats
                    merged_stats = counts.merge(price_stats[[col, 'Min Price', 'Max Price', 'Avg Price']], on=col)
                    
                    # Display as a dataframe
                    st.dataframe(merged_stats, use_container_width=True)
                    
                    # Create a horizontal bar chart using Plotly
                    fig = px.bar(
                        counts, 
                        y=col, 
                        x='Count', 
                        orientation='h',
                        color=col,
                        title=f"Distribution of {col}",
                        text='Percentage',
                        height=400
                    )
                    
                    fig.update_layout(
                        xaxis_title="Count",
                        yaxis_title=col,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display the full dataframe
        st.subheader("Full Dataset")
        st.dataframe(self.df)

        st.subheader("Data Summary")
        st.write(self.df.describe())

    def show_price_analysis(self):
        """Display price analysis page with interactive Plotly charts"""
        st.header(self.PAGE_PRICE_ANALYSIS)
        
        # Price distribution by company
        st.subheader("Price Distribution Across Companies")
        
        # Create a box plot for price distribution by company
        fig = px.box(self.df, x="Company", y="Price", 
                    title="Paddle Price Distribution by Company",
                    color="Company",
                    points="all",  # Show all points
                    hover_data=["Paddle"])  # Show paddle name on hover
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_title="Company",
            yaxis_title="Price ($)",
            xaxis_tickangle=-45,
            height=700,  # Taller chart
            showlegend=False
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Also add a histogram of overall price distribution
        hist_fig = px.histogram(
            self.df, 
            x="Price", 
            color="Company",
            title="Price Distribution Histogram",
            marginal="box",  # Add a box plot on the margin
            hover_data=["Paddle"]  # Show paddle name on hover
        )
        
        hist_fig.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Count",
            height=600  # Taller chart
        )
        
        st.plotly_chart(hist_fig, use_container_width=True)
        
        # Add price statistics
        st.subheader("Price Statistics by Company")
        
        # Group by company and calculate price statistics
        price_stats = self.df.groupby('Company')['Price'].agg([
            ('Average Price', 'mean'),
            ('Median Price', 'median'),
            ('Min Price', 'min'),
            ('Max Price', 'max'),
            ('Price Range', lambda x: x.max() - x.min()),
            ('Count', 'count')
        ]).reset_index().sort_values('Average Price', ascending=False)
        
        # Format the price columns - round to nearest integer
        for col in price_stats.columns:
            if col != 'Company' and col != 'Count':
                price_stats[col] = price_stats[col].apply(lambda x: int(round(x)))
        
        # Display the statistics
        st.dataframe(price_stats, use_container_width=True)

    def show_correlation_analysis(self):
        """Display correlation analysis page"""
        st.header(self.PAGE_CORRELATION)

        # Feature selection for correlation
        features = st.multiselect(
            "Select features for correlation matrix",
            options=self.numeric_cols,
            default=self.numeric_cols[:5]
        )

        if features:
            corr_fig = self.plot_correlation_matrix(features)
            st.pyplot(corr_fig, use_container_width=False)

        # Feature selection for scatter plot
        st.subheader("Explore Feature Relationships")
        col1, col2, col3 = st.columns(3)

        with col1:
            x_axis = st.selectbox("X-axis", options=self.numeric_cols)

        with col2:
            y_axis = st.selectbox("Y-axis", options=self.numeric_cols,
                                  index=1 if len(self.numeric_cols) > 1 else 0)

        with col3:
            color_options = ["None"] + self.categorical_cols + self.numeric_cols
            color_by = st.selectbox("Color by", options=color_options)
            if color_by == "None":
                color_by = None

        scatter_fig = self.plot_scatter(x_axis, y_axis, color_by)
        st.pyplot(scatter_fig, use_container_width=False)

    def show_feature_distribution(self):
        """Display feature distribution page"""
        st.header(self.PAGE_DISTRIBUTION)

        # Select feature to visualize
        dist_feature = st.selectbox(
            "Select feature to visualize",
            options=self.numeric_cols
        )

        if dist_feature:
            st.subheader(f"Distribution of {dist_feature}")
            
            # Create a Plotly histogram with KDE
            hist_fig = px.histogram(
                self.df, 
                x=dist_feature,
                title=f"Distribution of {dist_feature}",
                marginal="box",  # Add a box plot on the margin
                height=500
            )
            
            hist_fig.update_layout(
                xaxis_title=dist_feature,
                yaxis_title="Count",
                bargap=0.1  # Add some gap between bars
            )
            
            # Add a KDE curve
            hist_fig.update_traces(opacity=0.7)
            
            # Display the plot
            st.plotly_chart(hist_fig, use_container_width=True)

            # Categorical analysis
            st.subheader(f"{dist_feature} by Category")
            
            # Select categorical feature
            cat_feature = st.selectbox(
                "Select categorical feature",
                options=self.categorical_cols
            )
            
            if cat_feature:
                # Count values in each category
                cat_counts = self.df[cat_feature].value_counts()
                valid_cats = cat_counts[cat_counts >= 3].index.tolist()
                
                if valid_cats:
                    # Filter data to only include categories with enough data
                    filtered_data = self.df[self.df[cat_feature].isin(valid_cats)]
                    
                    # Create a Plotly box plot
                    box_fig = px.box(
                        filtered_data,
                        x=cat_feature,
                        y=dist_feature,
                        color=cat_feature,
                        title=f"{dist_feature} by {cat_feature}",
                        points="all",  # Show all data points
                        height=600
                    )
                    
                    box_fig.update_layout(
                        xaxis_title=cat_feature,
                        yaxis_title=dist_feature,
                        xaxis_tickangle=-45,
                        showlegend=False
                    )
                    
                    # Display the plot
                    st.plotly_chart(box_fig, use_container_width=True)
                else:
                    st.warning(f"Not enough data in categories of {cat_feature} for meaningful analysis.")

    def show_paddle_comparison(self):
        """Display paddle comparison page"""
        st.header(self.PAGE_COMPARISON)

        # Get all unique companies
        companies = sorted(self.df['Company'].unique())

        # First select company
        selected_companies = st.multiselect(
            "Select companies",
            options=companies
        )

        if selected_companies:
            # Get all paddles from selected companies
            filtered_df = self.df[self.df['Company'].isin(selected_companies)]
            available_paddles = sorted(filtered_df['Paddle'].unique())

            # Then select paddles from those companies
            paddles = st.multiselect(
                "Select paddles to compare",
                options=available_paddles
            )

            if paddles:
                # Define default features for comparison
                default_features = [
                    'Spin RPM',
                    'Twist Weight',
                    'Balance Point (cm)',
                    'Swing Weight',
                    'Punch Volley Speed-MPH (Pop)',
                    'Firepower (0-100)'
                ]

                # Filter default features to only include those that exist in the dataframe
                available_default_features = [f for f in default_features if f in self.numeric_cols]

                # Select features to compare
                comparison_features = st.multiselect(
                    "Select features to compare",
                    options=self.numeric_cols,
                    default=available_default_features
                )

                if comparison_features:
                    comparison_df = self.compare_paddles(paddles, comparison_features)

                    # Display comparison table
                    st.subheader("Comparison Table")
                    st.dataframe(comparison_df[['Paddle'] + comparison_features])

                    # Create radar charts
                    st.subheader("Radar Chart Comparison")
                    
                    # Create two columns for side-by-side display
                    col1, col2 = st.columns(2)
                    
                    # First radar chart in left column
                    with col1:
                        paddle_values = {}
                        for paddle in paddles:
                            values = {}
                            for f in comparison_features:
                                values[f] = self.df[self.df['Paddle'] == paddle][f].values[0]
                            paddle_values[paddle] = values

                        radar_fig = self.create_radar_chart(paddle_values, title="Performance Metrics")
                        st.pyplot(radar_fig, use_container_width=True)

                    # Second radar chart in right column
                    with col2:
                        percentile_features = [
                            'Price',
                            'Spin Percentile',
                            'Swing Weight Percentile',
                            'Twist Weight Percentile',
                            'Balance Point Percentile',
                            'Power Percentile',
                            'Pop Percentile'
                        ]
                        
                        # Filter to only include percentile features that exist in the dataframe
                        available_percentile_features = [f for f in percentile_features if f in self.df.columns]
                        
                        if available_percentile_features:
                            paddle_percentile_values = {}
                            for paddle in paddles:
                                values = {}
                                for f in available_percentile_features:
                                    if paddle in self.df['Paddle'].values:
                                        # Get the value and handle NaN
                                        value = self.df[self.df['Paddle'] == paddle][f].values[0]
                                        values[f] = value if not pd.isna(value) else 0
                                # Only add if we have values
                                if values:
                                    paddle_percentile_values[paddle] = values
                            
                            if paddle_percentile_values:
                                radar_percentile_fig = self.create_radar_chart(paddle_percentile_values, feature_order=available_percentile_features, title="Percentile & Price Comparison")
                                st.pyplot(radar_percentile_fig, use_container_width=True)
                            else:
                                st.warning("No percentile data available for the selected paddles.")
                        else:
                            st.warning("No percentile features available in the dataset.")

    def show_custom_analysis(self):
        """Display custom analysis page with filtering options"""
        st.header(self.PAGE_CUSTOM)

        # Filtering options
        st.subheader("Filter Data")
        filter_cat = st.selectbox(
            "Filter by category",
            options=["None"] + self.categorical_cols
        )

        if filter_cat != "None":
            filter_options = ["All"] + list(self.df[filter_cat].unique())
            filter_value = st.selectbox(
                f"Select {filter_cat} value",
                options=filter_options
            )

            if filter_value == "All":
                filter_value = None

            filtered_df = self.filter_data(filter_cat, filter_value)
        else:
            filtered_df = self.df

        st.write(f"Filtered data contains {len(filtered_df)} rows")
        st.dataframe(filtered_df)

        # Custom visualization options based on filtered data
        # ...


# Initialize and run app
if __name__ == "__main__":
    analysis_app = PaddleAnalysis()
    analysis_app.run_app()
