import streamlit as st
import pandas as pd
from chart_builders import create_charts, create_multi_oracle_comparison



# five comparisons we need to load the data



st.set_page_config(layout="wide")



eth_usd_combined_df = pd.read_csv('combined_df_ETH_USD_2025-01-22_17-06-11.csv')
kraken_eth_usd_prices_df = pd.read_csv('binance_df_ETH_USDT_2025-01-23_18-47-13.csv')

usdt_usd_combined_df = pd.read_csv('combined_df_USDT_USD_2025-01-22_17-10-00.csv')
kraken_usdt_usd_prices_df = pd.read_csv('kraken_df_USDT_USD_2025-01-22_17-09-55.csv')

usdc_usd_combined_df = pd.read_csv('combined_df_USDC_USD_2025-01-22_17-13-04.csv')
kraken_usdc_usd_prices_df = pd.read_csv('kraken_df_USDC_USD_2025-01-22_17-13-03.csv')

wbtc_usdt_combined_df = pd.read_csv('combined_df_WBTC_USDT_2025-01-22_17-18-49.csv')
binance_wbtc_usdt_prices_df = pd.read_csv('binance_df_WBTC_USDT_2025-01-22_17-18-46.csv')

zrc_usdt_combined_df = pd.read_csv('combined_df_ZRC_USDT_2025-01-22_17-32-25.csv')
bybit_zrc_usdt_prices_df = pd.read_csv('bybit_df_ZRC_USDT_2025-01-22_17-32-02.csv')




st.title("Price Comparison")
st.subheader("From Scroll Network (Mostly)")
st.markdown("---")
st.markdown("Context:")
st.markdown("We compare four major price feed providers — Chainlink, Redstone, Api3, and eOracle")
st.markdown("----")
st.markdown('Objectives of this research:')
st.markdown('Determine how eOracle objectively compares to competitors in terms of price accuracy based on deviation threshold settings during times of volatility')
st.markdown("----")
st.markdown("Why This Comparison Matters")
st.markdown("Latency and frequency of updates impact:")
st.markdown("    - User experience: Faster price resolution is ideal, especially in volatile conditions.")
st.markdown("    - Cost-effectiveness: Excessive updates waste resources.")
st.markdown("    - Price accuracy: Missed price changes can cause financial loss.")
st.markdown("----")
st.markdown("Competitors in this space use different heartbeat intervals and price thresholds.")
st.markdown("----")
st.markdown("Understanding their behavior helps us:")
st.markdown("    - Optimize for cost.")
st.markdown("    - Evaluate reliability.")
st.markdown("    - Identify competitive advantages.")


comparisons_dict = {
     'ETH-USD': {
        'combined_df': eth_usd_combined_df,
        'exchange_data_df': kraken_eth_usd_prices_df
     },
     'USDT-USD': {
        'combined_df': usdt_usd_combined_df,
        'exchange_data_df': kraken_usdt_usd_prices_df
     },
     'USDC-USD': {
        'combined_df': usdc_usd_combined_df,
        'exchange_data_df': kraken_usdc_usd_prices_df
     },
     'WBTC-USD': {
        'combined_df': wbtc_usdt_combined_df,
        'exchange_data_df': binance_wbtc_usdt_prices_df
     },
     'ZRC-USD (Zircuit Network)': {
        'combined_df': zrc_usdt_combined_df,
        'exchange_data_df': bybit_zrc_usdt_prices_df
     }

}

def app():
    try:
        # Sidebar selection
        st.sidebar.title('Select a comparison')
        st.sidebar.markdown("---")
        # available comparisons
        st.sidebar.markdown("Available comparisons:")
        for comparison in comparisons_dict.keys():
            st.sidebar.markdown(f"- {comparison}")
        st.sidebar.markdown("---")
        selected_comparison = st.sidebar.selectbox('Select a comparison', list(comparisons_dict.keys()))
        # if clicks button
        if st.sidebar.button('Display'):
            combined_df = comparisons_dict[selected_comparison]['combined_df']
            prices_df = comparisons_dict[selected_comparison]['exchange_data_df']

            # Create tabs for charts
            chart_tabs = st.tabs(["Time Series", "Distribution Analysis", "Update Analysis", "Interpolation Analysis", "Data Tables"])

            # Generate and display charts
            time_series_fig, dist_fig, bubble_figs = create_charts(combined_df, prices_df )

            if time_series_fig and dist_fig:
                with chart_tabs[0]:
                    st.plotly_chart(time_series_fig, use_container_width=True)
                with chart_tabs[1]:
                    st.plotly_chart(dist_fig, use_container_width=True)
                with chart_tabs[2]:
                    for feed, fig in bubble_figs.items():
                        st.plotly_chart(fig, use_container_width=True)
                with chart_tabs[3]:
                    pair_results = create_multi_oracle_comparison(combined_df)
                    if pair_results:
                        # Display each comparison figure and its statistics
                        for result in pair_results:
                            st.subheader(f"{result['oracle1']} vs {result['oracle2']} Comparison")
                            st.plotly_chart(result['figure'], use_container_width=True)

                            # Display statistics for this pair
                            st.markdown("**Pair Statistics:**")
                            st.dataframe(result['stats'], use_container_width=True)

                            # Add a divider between pairs
                            st.markdown("---")
                    else:
                        st.warning("Insufficient data for interpolation analysis")
                with chart_tabs[4]:
                # Display data tables
                    st.subheader("Data Tables")
            
                    dfs = {
                        'combined_df': combined_df,
                        'exchange_data_df': prices_df
                    }
                    st.dataframe(dfs['combined_df'], use_container_width=True)
                    st.dataframe(dfs['exchange_data_df'], use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    app()
