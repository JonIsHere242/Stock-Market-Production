import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numba import njit

# Import logging from Util
from Util import get_logger, dprint, Colors

# Initialize logger
logger = get_logger("AssetCorrelator")


def process_data_directory(data_dir='Data/RFpredictions'):
    """Process all Parquet files in the data directory."""
    logger.info(f"Starting asset correlation analysis from {data_dir}")
    
    # Load Parquet files into data_frames dictionary
    data_frames = {}
    files = [file for file in os.listdir(data_dir) if file.endswith('.parquet')]
    logger.info(f"Found {len(files)} parquet files to process")
    
    for file in tqdm(files, desc="Loading Parquet files"):
        ticker = file.split('.')[0]
        data_frames[ticker] = pd.read_parquet(os.path.join(data_dir, file))
    
    logger.debug(f"Loaded {len(data_frames)} ticker dataframes")
    
    # Process all stocks
    for ticker, df in tqdm(data_frames.items(), desc="Processing stocks"):
        data_frames[ticker] = process_stock_data(df)
    
    # Calculate returns and correlation matrix
    logger.info("Calculating weighted returns and correlation matrix")
    returns_df = pd.DataFrame({ticker: df['Weighted_Return'] for ticker, df in data_frames.items()})
    
    # Handle NaN values in returns_df - use backward fill and then forward fill
    returns_df = returns_df.fillna(method='bfill').fillna(method='ffill')
    
    # Check if we still have NaNs
    nan_count = returns_df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values after fillna. Dropping columns with NaN.")
        # Get columns with NaN
        cols_with_nan = returns_df.columns[returns_df.isna().any()].tolist()
        logger.debug(f"Columns with NaN: {cols_with_nan}")
        # Drop columns with NaN
        returns_df = returns_df.dropna(axis=1)
        logger.info(f"Dropped {len(cols_with_nan)} columns with NaN values. Remaining columns: {returns_df.shape[1]}")
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    
    # Handle NaN values in correlation matrix
    correlation_matrix = correlation_matrix.fillna(method='bfill').fillna(method='ffill')
    
    # Check if we still have NaNs in correlation matrix
    nan_count = correlation_matrix.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in correlation matrix. Dropping problematic rows/columns.")
        # Drop rows and columns with NaN
        correlation_matrix = correlation_matrix.dropna(axis=0).dropna(axis=1)
        logger.info(f"Correlation matrix shape after dropping NaNs: {correlation_matrix.shape}")
    
    # Final check for any remaining NaNs
    if correlation_matrix.isna().any().any():
        logger.error("Still have NaNs in correlation matrix. Replacing with zeros.")
        correlation_matrix = correlation_matrix.fillna(0)
    
    # Perform clustering with optimal cluster count
    logger.info("Standardizing correlation matrix")
    scaler = StandardScaler()
    scaled_correlation = scaler.fit_transform(correlation_matrix)
    
    # Find optimal number of clusters using silhouette score
    optimal_clusters = find_optimal_clusters(scaled_correlation)
    logger.info(f"Optimal number of clusters determined: {optimal_clusters}")
    
    # Run final clustering with optimal cluster count
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(scaled_correlation)
    
    # Create and process final results
    correlation_matrix_df = pd.DataFrame(correlation_matrix)
    correlation_matrix_df['Cluster'] = clusters
    
    clustered_assets = correlation_matrix_df[['Cluster']].reset_index()
    clustered_assets.columns = ['Ticker', 'Cluster']
    
    # Calculate group correlations
    group_correlations = calculate_group_correlations_vectorized(correlation_matrix, clustered_assets)
    
    # Final asset clustering with correlation metrics
    clustered_assets = process_clustered_assets(clustered_assets, group_correlations)
    
    # Save results to parquet
    output_file = 'Correlations.parquet'
    clustered_assets.to_parquet(output_file, index=False)
    logger.info(f"Correlations saved to '{output_file}'")
    
    # Debug print the ticker clusters
    dprint(f"Assets grouped into {optimal_clusters} clusters", level="INFO")
    for cluster_id in range(optimal_clusters):
        cluster_assets = clustered_assets[clustered_assets['Cluster'] == cluster_id]['Ticker'].tolist()
        dprint(f"Cluster {cluster_id}: {len(cluster_assets)} assets", level="DETAIL", indent=2)
    
    return clustered_assets


@njit
def calculate_dynamic_weights(volatility, mean_volatility, std_volatility):
    """Calculate dynamic weights based on volatility z-score."""
    if std_volatility == 0:
        return np.array([0.25, 0.25, 0.25, 0.25])
    
    z_score = (volatility - mean_volatility) / std_volatility
    weights = np.array([0.25 - 0.1*z_score, 0.25 - 0.05*z_score, 0.25 + 0.05*z_score, 0.25 + 0.1*z_score])
    weights = np.maximum(np.minimum(weights, 1), 0)  # This replaces np.clip
    return weights / np.sum(weights)


@njit
def calculate_weighted_returns_fast(returns, volatility, mean_volatility, std_volatility):
    """Calculate weighted returns using dynamic weights."""
    weighted_returns = np.zeros(len(returns))
    for i in range(21, len(returns)):
        weights = calculate_dynamic_weights(volatility[i], mean_volatility[i], std_volatility[i])
        weighted_returns[i] = np.sum(returns[i] * weights)
    return weighted_returns


def process_stock_data(df):
    """Process individual stock data to calculate weighted returns."""
    # Make copy to avoid modifying original
    df = df.copy()
    
    # Check for missing values in Close
    if df['Close'].isna().any():
        logger.debug(f"Found {df['Close'].isna().sum()} NaN values in Close column. Filling with forward fill then backward fill.")
        df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
    
    # Calculate returns with careful handling of NaN values
    df['Daily_Return'] = df['Close'].pct_change().fillna(method='bfill').round(3)
    df['Weekly_Return'] = df['Close'].pct_change(5).fillna(method='bfill').round(3)
    df['Monthly_Return'] = df['Close'].pct_change(21).fillna(method='bfill').round(3)
    df['Yearly_Return'] = df['Close'].pct_change(252).fillna(method='bfill').round(3)

    # Calculate volatility metrics
    df['Volatility'] = df['Daily_Return'].rolling(window=21).std().fillna(method='bfill')
    df['Mean_Volatility'] = df['Volatility'].rolling(window=21).mean().fillna(method='bfill')
    df['Volatility_Std'] = df['Volatility'].rolling(window=21).std().fillna(method='bfill')

    # Ensure no NaN values remain in these columns
    for col in ['Daily_Return', 'Weekly_Return', 'Monthly_Return', 'Yearly_Return', 'Volatility', 'Mean_Volatility', 'Volatility_Std']:
        if df[col].isna().any():
            logger.debug(f"Found NaN values in {col} after initial processing. Applying additional fill.")
            df[col] = df[col].fillna(method='ffill').fillna(0)  # Fill remaining NaNs with 0

    # Create the numpy arrays for weighted return calculation
    returns = np.column_stack((df['Daily_Return'], df['Weekly_Return'], df['Monthly_Return'], df['Yearly_Return']))
    volatility = df['Volatility'].values
    mean_volatility = df['Mean_Volatility'].values
    std_volatility = df['Volatility_Std'].values

    # Calculate weighted returns
    df['Weighted_Return'] = calculate_weighted_returns_fast(returns, volatility, mean_volatility, std_volatility)
    df['Weighted_Return'] = df['Weighted_Return'].round(3)
    
    # Final check for NaN values in weighted return
    if df['Weighted_Return'].isna().any():
        logger.warning(f"Found NaN values in Weighted_Return. Filling with zeros.")
        df['Weighted_Return'] = df['Weighted_Return'].fillna(0)

    return df


def find_optimal_clusters(scaled_correlation, min_clusters=6, max_clusters=19):
    """Find optimal number of clusters using silhouette score.
    
    Ensures a minimum of 4 clusters to match the trading strategy requirements.
    """
    logger.info("Finding optimal number of clusters")
    
    # Ensure no NaN or inf values in scaled_correlation
    if np.isnan(scaled_correlation).any() or np.isinf(scaled_correlation).any():
        logger.warning("Found NaN or Inf values in scaled correlation matrix. Replacing with zeros.")
        scaled_correlation = np.nan_to_num(scaled_correlation, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Adjust max_clusters if we have too few samples
    n_samples = scaled_correlation.shape[0]
    if n_samples < max_clusters:
        logger.warning(f"Reducing max_clusters from {max_clusters} to {n_samples-1} due to small sample size")
        max_clusters = max(min_clusters + 1, n_samples - 1)
    
    # Ensure min_clusters is at least 4 for trading strategy requirements
    min_clusters = max(4, min_clusters)
    logger.info(f"Evaluating clusters from {min_clusters} to {max_clusters}")
    
    silhouette_scores = []
    try:
        for k in tqdm(range(min_clusters, max_clusters + 1), desc="Evaluating clusters"):
            try:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = kmeans.fit_predict(scaled_correlation)
                score = silhouette_score(scaled_correlation, labels)
                silhouette_scores.append(score)
                logger.debug(f"Cluster {k}: silhouette score = {score:.4f}")
            except Exception as e:
                logger.error(f"Error calculating silhouette score for k={k}: {str(e)}")
                silhouette_scores.append(-1)  # Use a negative score to indicate failure
    except Exception as e:
        logger.error(f"Error in cluster evaluation: {str(e)}")
        # Fallback to default clusters (using at least 4)
        default_clusters = max(4, min(n_samples // 2, 8))
        logger.warning(f"Using default of {default_clusters} clusters due to evaluation error")
        return default_clusters
    
    # Find best cluster count, ignoring any that failed (-1 score)
    valid_scores = [(i+min_clusters, score) for i, score in enumerate(silhouette_scores) if score >= 0]
    if not valid_scores:
        logger.error("No valid silhouette scores found. Using default clusters.")
        default_clusters = max(4, min(n_samples // 2, 8))
        return default_clusters
        
    optimal_clusters = max(valid_scores, key=lambda x: x[1])[0]
    
    # Log warnings if cluster count is outside reasonable range
    # (but we already ensure it's at least 4)
    if optimal_clusters > 15:
        logger.warning(f"Unusually high number of clusters detected: {optimal_clusters}")
    
    dprint(f"Optimal number of clusters: {optimal_clusters}", level="SUCCESS")
    dprint(f"Silhouette score: {silhouette_scores[optimal_clusters - min_clusters]:.4f}", level="DETAIL")
    
    return optimal_clusters


def calculate_group_correlations_vectorized(correlation_matrix, clustered_assets):
    """Calculate correlations between assets and clusters."""
    logger.debug("Calculating group correlations")
    
    corr_array = correlation_matrix.values
    unique_clusters = clustered_assets['Cluster'].unique()
    cluster_indices = {
        cluster: clustered_assets.index[clustered_assets['Cluster'] == cluster].tolist() 
        for cluster in unique_clusters
    }
    
    result = pd.DataFrame(
        index=correlation_matrix.index, 
        columns=[f'correlation_{cluster}' for cluster in unique_clusters]
    )
    
    for cluster in tqdm(unique_clusters, desc="Calculating Group Correlations"):
        indices = cluster_indices[cluster]
        cluster_correlations = corr_array[:, indices].mean(axis=1)
        result[f'correlation_{cluster}'] = cluster_correlations
    
    return result


def process_clustered_assets(clustered_assets, group_correlations):
    """Calculate final metrics for clustered assets."""
    logger.debug("Processing final cluster metrics")
    
    try:
        # Merge group correlations with clustered assets
        clustered_assets = clustered_assets.merge(group_correlations, left_on='Ticker', right_index=True)
        
        # Calculate mean intra-group correlation
        mean_intra_group_corr = clustered_assets.groupby('Cluster')[group_correlations.columns].mean().mean(axis=1)
        clustered_assets['mean_intragroup_correlation'] = clustered_assets['Cluster'].map(mean_intra_group_corr)
        
        # Calculate difference to mean group correlation
        clustered_assets['diff_to_mean_group_corr'] = clustered_assets.apply(
            lambda row: row[f'correlation_{row.Cluster}'] - row['mean_intragroup_correlation'], 
            axis=1
        )
        
        # Check for any NaN values before final processing
        if clustered_assets.isna().any().any():
            logger.warning("Found NaN values in final metrics. Filling with appropriate values.")
            # Fill NaN values in diff_to_mean_group_corr with 0
            clustered_assets['diff_to_mean_group_corr'] = clustered_assets['diff_to_mean_group_corr'].fillna(0)
            # Fill NaN values in mean_intragroup_correlation with column mean
            clustered_assets['mean_intragroup_correlation'] = clustered_assets['mean_intragroup_correlation'].fillna(
                clustered_assets['mean_intragroup_correlation'].mean()
            )
            # Fill NaN values in correlation columns with 0
            for col in group_correlations.columns:
                clustered_assets[col] = clustered_assets[col].fillna(0)
        
        # Reorder columns and round values
        reordered_columns = [
            'Ticker', 'Cluster', 'mean_intragroup_correlation', 
            'diff_to_mean_group_corr'
        ] + list(group_correlations.columns)
        
        clustered_assets = clustered_assets[reordered_columns].round(5)
        
    except Exception as e:
        logger.error(f"Error in process_clustered_assets: {str(e)}")
        # Return a simplified version if processing fails
        clustered_assets = clustered_assets[['Ticker', 'Cluster']]
        
    return clustered_assets





if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Asset Correlation Analysis')
    parser.add_argument('--data-dir', type=str, default='Data/RFpredictions', help='Directory containing parquet files')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--min-clusters', type=int, default=6, help='Minimum number of clusters to use (default: 6)')
    parser.add_argument('--max-clusters', type=int, default=19, help='Maximum number of clusters to evaluate (default: 19)')
    args = parser.parse_args()
    
    if args.debug:
        logger = get_logger("AssetCorrelator", debug=True)
        logger.debug("Debug mode enabled")
    
    logger.info(f"Using minimum of {args.min_clusters} clusters for asset correlation")
    
    try:
        # Run the analysis
        process_data_directory(args.data_dir)
    except Exception as e:
        logger.error(f"Fatal error in asset correlation analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())