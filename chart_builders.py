# chart_builders.py

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import interpolate

def create_charts(combined_df, binance_df=None):
    if combined_df.empty:
        return None, None, None
        
    combined_df['price_delta_abs'] = combined_df['price_delta'].abs()
    
    # Create main figure with 2 subplots
    fig1 = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1, 
        row_heights=[0.5, 0.5],
        subplot_titles=("Price Data", "Price Delta")
    )
    
    # Create distribution analysis figure
    fig2 = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        specs=[[{"type": "violin"}, None], 
               [{"type": "scatter"}, {"type": "violin"}]]
    )
    
    # Add Binance data if available
    if binance_df is not None and not binance_df.empty:
        min_time = combined_df['timestamp'].min()
        max_time = combined_df['timestamp'].max()
        binance_df_filtered = binance_df[
            (binance_df['timestamp'] >= min_time) & 
            (binance_df['timestamp'] <= max_time)
        ]
        
        fig1.add_trace(
            go.Candlestick(
                x=binance_df_filtered['timestamp'],
                open=binance_df_filtered['open'],
                high=binance_df_filtered['high'],
                low=binance_df_filtered['low'],
                close=binance_df_filtered['close'],
                name='Binance',
                opacity=0.3
            ),
            row=1, col=1
        )

    # Oracle colors
    oracle_colors = {
        'Chainlink': 'blue',
        'Redstone': 'red',
        'Api3': 'purple',
        'Eoracle': 'green'
    }
    
    # Create bubble charts dictionary
    bubble_figs = {}

    # Process each oracle's data
    for feed, color in oracle_colors.items():
        df = combined_df[combined_df['feed'] == feed].copy()
        if df.empty:
            continue
            
        # Create bubble chart
        df_filtered = df[df['time_delta'] > 0].copy()
        bubble_figs[feed] = px.scatter(
            df_filtered, 
            x='timestamp', 
            y='price', 
            size='time_delta',
            color='price_delta_abs',
            title=f"{feed} Updates (bubble size = time delta, color = |price delta|)",
            color_continuous_scale='Viridis'
        ).update_layout(
            template="plotly_dark",
            height=400,
            coloraxis_colorbar_title="|Price Δ| (%)"
        )
        
        # Add traces to main figure
        fig1.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                mode='lines+markers',
                name=feed,
                legendgroup=feed,
                line=dict(color=color, width=2),
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.6,
                    line=dict(width=2, color='DarkSlateGrey')
                )
            ),
            row=1, col=1
        )
        
        fig1.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['price_delta'],
                mode='lines+markers',
                name=feed,
                legendgroup=feed,
                line=dict(color=color, width=2),
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.6,
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add distribution analysis traces
        fig2.add_trace(
            go.Scatter(
                x=df['price_delta'],
                y=df['time_delta'],
                mode='markers',
                name=feed,
                marker=dict(color=color)
            ),
            row=2, col=1
        )
        
        fig2.add_trace(
            go.Violin(
                y=df['time_delta'],
                name=f"{feed} Time Δ",
                marker=dict(color=color),
                side='positive',
                box_visible=True,
                meanline_visible=True
            ),
            row=2, col=2
        )
        
        fig2.add_trace(
            go.Violin(
                x=df['price_delta'],
                name=f"{feed} Price Δ",
                marker=dict(color=color),
                side='positive',
                box_visible=True,
                meanline_visible=True
            ),
            row=1, col=1
        )
    
    # Update layouts
    fig1.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title="Price and Price Delta Comparison",
        yaxis_title="Price (USD)",
        yaxis2_title="Price Delta (%)",
        xaxis_rangeslider_visible=False
    )
    
    fig2.update_layout(
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title="Distribution Analysis",
        xaxis_title="Price Delta (%)",
        yaxis_title="Time Delta (seconds)"
    )
    
    return fig1, fig2, bubble_figs

def create_interpolation_charts(combined_df, oracle1='Eoracle', oracle2='Api3'):
    if combined_df.empty:
        return None, None, None, None
    # Get min and max block numbers
    min_block = combined_df['block_number'].min()
    max_block = combined_df['block_number'].max()
    all_blocks = pd.DataFrame({'block_number': range(int(min_block), int(max_block)+1)})
    
    # Split the data by selected feeds
    df_oracle1_temp = combined_df[combined_df['feed'] == oracle1]
    df_oracle2_temp = combined_df[combined_df['feed'] == oracle2]

    if df_oracle1_temp.empty or df_oracle2_temp.empty:
        return None, None, None, None

    # Define interpolation methods
    interp_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']
    
    # Create figure for all interpolation methods
    fig1 = make_subplots(
        rows=len(interp_methods), 
        cols=1,
        subplot_titles=[f"{method.capitalize()} Interpolation" for method in interp_methods],
        vertical_spacing=0.03
    )

    # Dictionary to store results for each method
    all_results = {}

    for idx, method in enumerate(interp_methods, 1):
        # Create interpolation functions for each feed
        oracle1_interp = interpolate.interp1d(df_oracle1_temp['block_number'], 
                                            df_oracle1_temp['price'],
                                            kind=method,
                                            fill_value='extrapolate')

        oracle2_interp = interpolate.interp1d(df_oracle2_temp['block_number'], 
                                            df_oracle2_temp['price'],
                                            kind=method,
                                            fill_value='extrapolate')

        # Create continuous data for each feed
        all_blocks[f'{oracle1.lower()}_{method}_price'] = oracle1_interp(all_blocks['block_number'])
        all_blocks[f'{oracle2.lower()}_{method}_price'] = oracle2_interp(all_blocks['block_number'])

        # Calculate differences
        all_blocks[f'price_difference_{method}'] = (
            all_blocks[f'{oracle1.lower()}_{method}_price'] - 
            all_blocks[f'{oracle2.lower()}_{method}_price']
        )

        # Add traces to the subplot
        fig1.add_trace(
            go.Scatter(
                x=df_oracle1_temp['block_number'],
                y=df_oracle1_temp['price'],
                mode='markers',
                name=f'{oracle1} Original',
                marker=dict(size=4),
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )

        fig1.add_trace(
            go.Scatter(
                x=df_oracle2_temp['block_number'],
                y=df_oracle2_temp['price'],
                mode='markers',
                name=f'{oracle2} Original',
                marker=dict(size=4),
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )

        fig1.add_trace(
            go.Scatter(
                x=all_blocks['block_number'],
                y=all_blocks[f'{oracle1.lower()}_{method}_price'],
                mode='lines',
                name=f'{oracle1} {method}',
                line=dict(width=1),
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )

        fig1.add_trace(
            go.Scatter(
                x=all_blocks['block_number'],
                y=all_blocks[f'{oracle2.lower()}_{method}_price'],
                mode='lines',
                name=f'{oracle2} {method}',
                line=dict(width=1),
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )
        # Store results for comparison
        all_results[method] = {
            'difference': all_blocks[f'price_difference_{method}'].describe(),
            'max_diff': all_blocks[f'price_difference_{method}'].abs().max(),
            'mean_diff': all_blocks[f'price_difference_{method}'].abs().mean()
        }

    # Update layout
    fig1.update_layout(
        template="plotly_dark",
        height=300 * len(interp_methods),
        title=f'Interpolation Methods Comparison ({oracle1} vs {oracle2})',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    # Create comparison statistics dataframe
    comparison_df = pd.DataFrame({
        method: {
            'Max Absolute Difference': results['max_diff'],
            'Mean Absolute Difference': results['mean_diff']
        }
        for method, results in all_results.items()
    }).T

    # Create detailed statistics for the selected method
    stats_df = pd.concat(
        {method: results['difference'] for method, results in all_results.items()},
        axis=1
    )

    return fig1, comparison_df, stats_df, all_blocks

def create_multi_oracle_comparison(combined_df, interp_kind='previous'):
    """Compare multiple oracle combinations with separate figures and statistics"""
    if combined_df.empty:
        return None

    # Get all available oracles
    available_oracles = combined_df['feed'].unique()
    oracle_pairs = []
    pair_results = []  # Will store tuples of (figure, stats) for each pair
    
    # Create all possible oracle pairs
    for i in range(len(available_oracles)):
        for j in range(i + 1, len(available_oracles)):
            oracle_pairs.append((available_oracles[i], available_oracles[j]))

    for oracle1, oracle2 in oracle_pairs:
        # Create subplots for this pair
        fig = make_subplots(
            rows=3, 
            cols=1,
            subplot_titles=[
                f"{oracle1} vs {oracle2} Prices",
                f"Price Difference",
                f"Cumulative Difference"
            ],
            vertical_spacing=0.1
        )

        # Get data for each oracle
        df_oracle1 = combined_df[combined_df['feed'] == oracle1]
        df_oracle2 = combined_df[combined_df['feed'] == oracle2]

        if not df_oracle1.empty and not df_oracle2.empty:
            # Create interpolation functions
            oracle1_interp = interpolate.interp1d(
                df_oracle1['block_number'], 
                df_oracle1['price'],
                kind=interp_kind,
                fill_value='extrapolate'
            )

            oracle2_interp = interpolate.interp1d(
                df_oracle2['block_number'], 
                df_oracle2['price'],
                kind=interp_kind,
                fill_value='extrapolate'
            )

            # Create continuous data
            blocks = np.arange(
                int(min(df_oracle1['block_number'].min(), df_oracle2['block_number'].min())),
                int(max(df_oracle1['block_number'].max(), df_oracle2['block_number'].max())) + 1
            )
            
            prices1 = oracle1_interp(blocks)
            prices2 = oracle2_interp(blocks)
            
            # Calculate differences
            differences = prices1 - prices2

            # Create DataFrame for cumulative difference calculation
            diff_df = pd.DataFrame({
                'block_number': blocks,
                'price_difference': differences
            })
            
            # Merge with original data to get timestamps
            merged_df = pd.merge(
                diff_df,
                combined_df[['block_number', 'timestamp']].drop_duplicates(),
                on='block_number',
                how='left'
            )
            
            # Drop rows where timestamp is NaN and calculate cumulative difference
            merged_df = merged_df[merged_df['timestamp'].notna()]
            merged_df['cumulative_difference'] = merged_df['price_difference'].cumsum()

            # Add price traces
            fig.add_trace(
                go.Scatter(
                    x=df_oracle1['block_number'],
                    y=df_oracle1['price'],
                    mode='markers',
                    name=f'{oracle1} Original',
                    marker=dict(size=4)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df_oracle2['block_number'],
                    y=df_oracle2['price'],
                    mode='markers',
                    name=f'{oracle2} Original',
                    marker=dict(size=4)
                ),
                row=1, col=1
            )

            # Add interpolated price traces
            fig.add_trace(
                go.Scatter(
                    x=blocks,
                    y=prices1,
                    mode='lines',
                    name=f'{oracle1} Interpolated',
                    line=dict(width=1)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=blocks,
                    y=prices2,
                    mode='lines',
                    name=f'{oracle2} Interpolated',
                    line=dict(width=1)
                ),
                row=1, col=1
            )

            # Add difference trace
            fig.add_trace(
                go.Scatter(
                    x=blocks,
                    y=differences,
                    mode='lines',
                    name=f'Price Difference',
                    line=dict(width=1),
                    fill='tozeroy'
                ),
                row=2, col=1
            )

            # Add cumulative difference trace
            fig.add_trace(
                go.Scatter(
                    x=merged_df['block_number'],
                    y=merged_df['cumulative_difference'],
                    mode='lines',
                    name=f'Cumulative Difference',
                    line=dict(width=1),
                    fill='tozeroy'
                ),
                row=3, col=1
            )

            # Update y-axis titles
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Difference", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative Diff", row=3, col=1)

            # Update layout for this figure
            fig.update_layout(
                template="plotly_dark",
                height=900,  # Fixed height for each comparison
                title=f'{oracle1} vs {oracle2} Comparison ({interp_kind} interpolation)',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Calculate statistics for this pair
            pair_stats = {
                'Mean Difference': float(np.nanmean(merged_df['price_difference'])),
                'Std Difference': float(np.nanstd(merged_df['price_difference'])),
                'Max Absolute Difference': float(np.nanmax(np.abs(merged_df['price_difference']))),
                'Median Difference': float(np.nanmedian(merged_df['price_difference'])),
                'Final Cumulative Difference': float(merged_df['cumulative_difference'].iloc[-1])
            }
            
            # Convert to DataFrame for better display and format numbers
            stats_df = pd.DataFrame([pair_stats]).T
            stats_df.columns = ['Value']
            
            # Format numbers to 4 decimal places
            stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.4f}")

            # Store figure and stats together
            pair_results.append({
                'oracle1': oracle1,
                'oracle2': oracle2,
                'figure': fig,
                'stats': stats_df
            })
        
    return pair_results

def create_detailed_plots(all_blocks, method, oracle1, oracle2):
    """
    Create detailed analysis plots for a selected interpolation method.
    """
    # Create subplots for detailed analysis
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=[
            f"Price Comparison ({method} interpolation)",
            "Price Difference Distribution",
            "Cumulative Price Difference"
        ],
        vertical_spacing=0.1
    )

    # Price comparison
    fig.add_trace(
        go.Scatter(
            x=all_blocks['block_number'],
            y=all_blocks[f'{oracle1.lower()}_{method}_price'],
            mode='lines',
            name=f'{oracle1} {method}',
            line=dict(width=1.5)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=all_blocks['block_number'],
            y=all_blocks[f'{oracle2.lower()}_{method}_price'],
            mode='lines',
            name=f'{oracle2} {method}',
            line=dict(width=1.5)
        ),
        row=1, col=1
    )

    # Price difference distribution
    fig.add_trace(
        go.Histogram(
            x=all_blocks[f'price_difference_{method}'],
            name='Price Difference Distribution',
            nbinsx=50,
            opacity=0.7
        ),
        row=2, col=1
    )

    # Cumulative difference
    cumsum = all_blocks[f'price_difference_{method}'].cumsum()
    fig.add_trace(
        go.Scatter(
            x=all_blocks['block_number'],
            y=cumsum,
            mode='lines',
            name='Cumulative Difference',
            fill='tozeroy',
            line=dict(width=1.5)
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=900,
        title=f'Detailed Analysis for {method.capitalize()} Interpolation',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Update y-axes titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Difference", row=3, col=1)

    # Update x-axes titles
    fig.update_xaxes(title_text="Block Number", row=3, col=1)
    fig.update_xaxes(title_text="Price Difference", row=2, col=1)
    fig.update_xaxes(title_text="Block Number", row=1, col=1)

    # Add statistics annotations
    diff_series = all_blocks[f'price_difference_{method}']
    stats_text = (
        f"Statistics:<br>"
        f"Mean: {diff_series.mean():.6f}<br>"
        f"Std: {diff_series.std():.6f}<br>"
        f"Min: {diff_series.min():.6f}<br>"
        f"Max: {diff_series.max():.6f}<br>"
        f"Median: {diff_series.median():.6f}"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.0,
        y=0.95,
        text=stats_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="white",
        borderwidth=1
    )

    return fig