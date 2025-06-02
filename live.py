import streamlit as st
import yfinance as yf
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import datetime

def safe_metric(label, value, change=None, prefix="₹"):
    if value is None or (isinstance(value, float) and (np.isnan(value) or value == 0)):
        value_str = "N/A"
    else:
        value_str = f"{prefix}{value:,.2f}" if isinstance(value, (float, int)) else str(value)
    if change is not None:
        st.metric(label, value_str, change)
    else:
        st.metric(label, value_str)

def render_live_dashboard(stock, theme):
    # Watchlist in session state
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = []
    st.sidebar.markdown("**Watchlist**")
    add_stock = st.sidebar.text_input("Add Ticker to Watchlist", "")
    if st.sidebar.button("Add to Watchlist") and add_stock and add_stock not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(add_stock.upper())
    remove_stock = st.sidebar.selectbox("Remove from Watchlist", ["-"] + st.session_state['watchlist'])
    if st.sidebar.button("Remove Selected") and remove_stock != "-":
        st.session_state['watchlist'].remove(remove_stock)
    st.sidebar.write(st.session_state['watchlist'])

    # Auto-refresh every 60 seconds
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60 * 1000, key="refresh")

    # Robust data fetching
    def is_volume_empty(df):
        return df.empty or ('Volume' not in df.columns) or (df['Volume'].dropna().sum() == 0)

    # Try 1m, then 5m, then 1d intervals
    df_live = yf.download(stock, period="5d", interval="1m")
    if len(df_live) < 30 or is_volume_empty(df_live):
        df_live = yf.download(stock, period="1mo", interval="5m")
        st.info("No recent intraday data. Showing 5-min data instead.")
    if len(df_live) < 10 or is_volume_empty(df_live):
        df_live = yf.download(stock, period="6mo", interval="1d")
        st.info("No recent intraday data. Showing daily data instead.")

    if df_live.empty or is_volume_empty(df_live):
        st.error(
            """No live data available for this stock from Yahoo Finance.\n"
            "- Try again during Indian market hours (9:15am–3:30pm IST).\n"
            "- Try a different stock.\n"
            "- Or check your internet connection."""
        )
        return

    st.write(f"Rows in df_live: {len(df_live)}")
    st.dataframe(df_live.tail(10))

    ticker = yf.Ticker(stock)
    info = ticker.info
    market_cap = info.get('marketCap', None)
    pe_ratio = info.get('trailingPE', None)
    sector = info.get('sector', 'N/A')
    news = ticker.news if hasattr(ticker, 'news') else []

    if not df_live.empty:
        latest = df_live.iloc[-1]
        prev_close = float(df_live['Close'].iloc[-2]) if len(df_live) > 1 else None
        price = float(latest['Close']) if 'Close' in latest else None
        change = price - prev_close if price is not None and prev_close is not None else None
        pct_change = (change / prev_close) * 100 if change is not None and prev_close else None
        st.markdown(f"### Live Stock Dashboard: <span style='color:{theme['primary']}'>{stock}</span>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        safe_metric("Current Price", price, f"{change:+.2f} ({pct_change:+.2f}%)" if change is not None else None)
        safe_metric("Open", float(latest['Open']) if 'Open' in latest else None)
        safe_metric("High", float(latest['High']) if 'High' in latest else None)
        safe_metric("Low", float(latest['Low']) if 'Low' in latest else None)
        safe_metric("Prev Close", prev_close)
        safe_metric("Volume", int(latest['Volume']) if 'Volume' in latest and latest['Volume'] != 0 else None, prefix="")
        safe_metric("Market Cap", market_cap, prefix="₹" if market_cap else "")
        safe_metric("P/E Ratio", pe_ratio, prefix="" if pe_ratio else "")
        st.markdown(f"**Sector:** {sector}")
        # Technical Indicators
        if len(df_live) >= 26:
            df_live['RSI'] = ta.rsi(df_live['Close'], length=14)
            macd = ta.macd(df_live['Close'])
            if df_live['RSI'].notna().sum() > 0:
                st.markdown(f"#### RSI (14)")
                st.line_chart(df_live['RSI'], use_container_width=True)
            else:
                st.warning("RSI could not be calculated (not enough data).")
            if macd is not None and macd.notna().sum().sum() > 0:
                df_live = df_live.join(macd)
                st.markdown(f"#### MACD")
                if 'MACD_12_26_9' in df_live.columns and df_live['MACD_12_26_9'].notna().sum() > 0:
                    st.line_chart(df_live['MACD_12_26_9'], use_container_width=True)
                else:
                    st.warning("MACD could not be calculated (not enough data).")
            else:
                st.warning("MACD could not be calculated (not enough data).")
        else:
            st.warning("Not enough data for RSI/MACD. These indicators require at least 26 data points. Try a different stock, interval, or wait for more data to accumulate.")
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df_live.index,
            open=df_live['Open'] if 'Open' in df_live else [],
            high=df_live['High'] if 'High' in df_live else [],
            low=df_live['Low'] if 'Low' in df_live else [],
            close=df_live['Close'] if 'Close' in df_live else [],
            increasing_line_color=theme['secondary'], decreasing_line_color=theme['danger']
        )])
        fig.update_layout(
            title=f"{stock} Live Candlestick Chart",
            xaxis_title="Time",
            yaxis_title="Price (INR)",
            plot_bgcolor=theme['plot_bg'],
            paper_bgcolor=theme['paper_bg'],
            font=dict(color=theme['font'])
        )
        st.plotly_chart(fig, use_container_width=True)
        # News Feed
        st.markdown(f"#### Latest News")
        if news:
            for item in news[:5]:
                title = item.get('title')
                link = item.get('link')
                if title and link:
                    st.write(f"[{title}]({link})")
        else:
            st.write("No news available.")
    else:
        st.error("No live data available for this stock.") 