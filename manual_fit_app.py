import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def model_ui_curve(x, a, b, c, t):
    return t + a * x + b * np.log10(x / c)


def calculate_r2_and_residuals(y_meas: pd.Series, y_fit: pd.Series):
    residuals = y_meas - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_meas - np.mean(y_meas))**2)
    r2 = 1 - ss_res / ss_tot
    return r2, residuals


# -- Streamlit page setup --
st.set_page_config(layout="wide")

# Initialize session_state defaults
if 'resistance' not in st.session_state:
    st.session_state.resistance = 250       # mΩ·cm²
if 'tafel' not in st.session_state:
    st.session_state.tafel = 0.200          # V/dec
if 'exchange' not in st.session_state:
    st.session_state.exchange = 0.50        # mA/cm²
# Max bounds from sliders
max_a_slider = 2000
max_a = max_a_slider / 1e6
max_b = 0.400
max_c = 5.00

# -- Data input --
empty = pd.DataFrame({
    "current dens. [mA/cm^2]": [],
    "voltage - measured [V]": []
})
data_df = st.data_editor(empty, num_rows='dynamic', key='data_editor')
st.session_state.data_df = data_df.dropna(ignore_index=True)

# -- Layout --
col1, col2 = st.columns([0.3, 0.7])
with col1:
    # Data and constants
    markers = st.checkbox("Show markers!", value=False)
    t_val = st.number_input("Insert value of constant t", value=1.19, key='t_val')

    # Parameter sliders
    resistance = st.slider(
        "Resistance in milliOhm.cm^2",
        min_value=1,
        max_value=max_a_slider,
        step=1,
        key='resistance'
    )
    tafel = st.slider(
        "Tafel Slope in V/dec",
        min_value=0.001,
        max_value=max_b,
        step=0.001,
        format="%.3f",
        key='tafel'
    )
    exchange = st.slider(
        "Exchange current in mA/cm^2",
        min_value=0.001,
        max_value=max_c,
        step=0.001,
        format="%.3f",
        key='exchange'
    )

    # Show current fit statistics
    df = st.session_state.data_df
    if not df.empty:
        sim_vals = model_ui_curve(
            df["current dens. [mA/cm^2]"],
            st.session_state.resistance/1e6,
            st.session_state.tafel,
            st.session_state.exchange,
            st.session_state.t_val
        )
        r2, res = calculate_r2_and_residuals(df["voltage - measured [V]"], sim_vals)
        subcol1, subcol2, subcol3 = st.columns([2,3,3])
        with subcol1:
            st.write(f"**Current fit:**")
        with subcol2:
            if r2 > 0.9:
                st.write(f"R²: :blue-background[**{r2:.6f}**]")
            else:
                st.write(f"R²: :blue-background[:red[**{r2:.6f}**]]")
        with subcol3:
            st.write(f"SSR×10⁶: :blue-background[**{np.sum(res**2)*1e6:.2f}**]")

    # Lock options at top
    lock_opt = st.radio(
        "Lock value of following variable(s):",
        [
            "None",
            "Resistance",
            "Tafel Slope",
            "Exchange current",
            "Resistance and Tafel",
            "Resistance and Exchange",
            "Tafel and Exchange"
        ],
        index=0,
        key='lock_opt'
    )

    # Fitting logic
    def fit_curve_callback():
        df = st.session_state.data_df
        if df.empty:
            return
        x = df["current dens. [mA/cm^2]"].values
        y = df["voltage - measured [V]"].values
        t = st.session_state.t_val

        # Current slider values
        a0 = st.session_state.resistance / 1e6
        b0 = st.session_state.tafel
        c0 = st.session_state.exchange

        opt_a, opt_b, opt_c = a0, b0, c0
        mode = st.session_state.lock_opt

        # Fit for each lock state
        if mode == "None":
            def f(x_, a, b, c): return model_ui_curve(x_, a, b, c, t)
            p0, bounds = [a0, b0, c0], ([0, 0, 0], [max_a, max_b, max_c])
        elif mode == "Resistance":
            def f(x_, b, c): return model_ui_curve(x_, a0, b, c, t)
            p0, bounds = [b0, c0], ([0, 0], [max_b, max_c])
        elif mode == "Tafel Slope":
            def f(x_, a, c): return model_ui_curve(x_, a, b0, c, t)
            p0, bounds = [a0, c0], ([0, 0], [max_a, max_c])
        elif mode == "Exchange current":
            def f(x_, a, b): return model_ui_curve(x_, a, b, c0, t)
            p0, bounds = [a0, b0], ([0, 0], [max_a, max_b])
        elif mode == "Resistance and Tafel":
            def f(x_, c): return model_ui_curve(x_, a0, b0, c, t)
            p0, bounds = [c0], ([0], [max_c])
        elif mode == "Resistance and Exchange":
            def f(x_, b): return model_ui_curve(x_, a0, b, c0, t)
            p0, bounds = [b0], ([0], [max_b])
        else:  # "Tafel and Exchange"
            def f(x_, a): return model_ui_curve(x_, a, b0, c0, t)
            p0, bounds = [a0], ([0], [max_a])

        popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=100000)
        # Map optimized values back
        if mode == "None": opt_a, opt_b, opt_c = popt
        elif mode == "Resistance": opt_b, opt_c = popt
        elif mode == "Tafel Slope": opt_a, opt_c = popt
        elif mode == "Exchange current": opt_a, opt_b = popt
        elif mode == "Resistance and Tafel": opt_c = popt[0]
        elif mode == "Resistance and Exchange": opt_b = popt[0]
        else: opt_a = popt[0]

        st.session_state.resistance = int(opt_a * 1e6)
        st.session_state.tafel = opt_b
        st.session_state.exchange = opt_c

    st.button("Fit curve", on_click=fit_curve_callback)

# -- Plotting in second column --
current_sim = np.linspace(
    5,
    st.session_state.data_df["current dens. [mA/cm^2]"].max()*1.1,
    1000
)
voltage_sim = model_ui_curve(
    current_sim,
    st.session_state.resistance/1e6,
    st.session_state.tafel,
    st.session_state.exchange,
    st.session_state.t_val
)
with col2:
    fig, ax = plt.subplots(figsize=(10,6))
    if markers and not st.session_state.data_df.empty:
        ax.scatter(
            st.session_state.data_df["current dens. [mA/cm^2]"],
            st.session_state.data_df["voltage - measured [V]"],
            marker='o', color='#FF0000', label='Measured'
        )
    elif not st.session_state.data_df.empty:
        ax.plot(
            st.session_state.data_df["current dens. [mA/cm^2]"],
            st.session_state.data_df["voltage - measured [V]"],
            linestyle='-', color='#FF0000', label='Measured'
        )
    ax.plot(current_sim, voltage_sim, linestyle='--', color='#0000FF', label='Simulated')
    ax.set_xlabel("Current Density [mA/cm²]")
    ax.set_ylabel("Voltage [V]")
    if not st.session_state.data_df.empty:
        ax.set_xlim(0, st.session_state.data_df["current dens. [mA/cm^2]"].max()*1.15)
        ax.set_ylim(
            st.session_state.data_df["voltage - measured [V]"].min()*0.95,
            st.session_state.data_df["voltage - measured [V]"].max()*1.1
        )
    ax.legend(loc='upper left')
    st.pyplot(fig)
