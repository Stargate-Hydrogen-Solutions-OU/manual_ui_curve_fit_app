import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def model_ui_curve(x, a, b, c, t):
    """
    Equation to be fitted.
    x is known, t is passed to the function as a global variable. Parameters a, b, and c are fitted
    """
    return t + a * x + b * np.log10(x / c)


def calculate_r2_and_residuals(y_measured: pd.Series, y_fit: pd.Series):
    """Calculate R² (coefficient of determination) and residuals."""
    residuals = y_measured - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_measured - np.mean(y_measured))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2, residuals


st.set_page_config(layout="wide")

empty_df = pd.DataFrame({"current dens. [mA/cm^2]": [], "voltage - measured [V]": []})
data_df = st.data_editor(
    data=empty_df,
    num_rows="dynamic",
)

col1, col2 = st.columns([0.3, 0.7], gap="small", vertical_alignment="top", border=False)

with col1:
    markers = st.checkbox("Show markers! (use for data with low amount of datapoints)")
    t_val = st.number_input("Insert value of constant t", value=1.19)

    resistance_a = st.slider(
        "Resistance in milliOhm.cm^2",
        min_value=10,
        max_value=2000,
        step=5,
        value=250,
        label_visibility="visible"
    )
    tafel_slope_b = st.slider(
        "Tafel Slope in V/dec",
        min_value=0.001,
        max_value=0.400,
        step=0.001,
        value=0.200,
        format="%.3f",
        label_visibility="visible"
    )
    exchange_curr_c = st.slider(
        "Exchange current in mA/cm^2",
        min_value=0.01,
        max_value=5.00,
        step=0.01,
        value=0.50,
        format="%.3f",
        label_visibility="visible"
    )
    data_df["simulated [V]"] = model_ui_curve(
        data_df["current dens. [mA/cm^2]"],
        resistance_a / 1000000,
        tafel_slope_b,
        exchange_curr_c,
        t_val
    )

    r2_res, residuals_arr = calculate_r2_and_residuals(
        data_df["voltage - measured [V]"],
        data_df["simulated [V]"]
    )
    st.write("Value of R^2 is:", round(r2_res, 6))
    st.write("Value of SSR * 10^6 is:", round(np.sum(residuals_arr ** 2) * 1e6, 2))


current_sim = np.linspace(5, data_df["current dens. [mA/cm^2]"].max()*1.1, 1000)

voltage_sim = model_ui_curve(
    current_sim,
    resistance_a/1000000,
    tafel_slope_b,
    exchange_curr_c,
    t_val
)


data_sim = pd.DataFrame(
    {"current dens. [mA/cm^2]": current_sim,
     "simulated [V]": voltage_sim
     }
)


with col2:
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot measured data
    if markers:
        ax.scatter(data_df["current dens. [mA/cm^2]"], data_df["voltage - measured [V]"], label="Measured",
                   marker="o", color="#FF0000")
    else:
        ax.plot(data_df["current dens. [mA/cm^2]"], data_df["voltage - measured [V]"], label="Measured",
                linestyle="-", color="#FF0000")

    # Plot simulated data
    ax.plot(data_sim["current dens. [mA/cm^2]"], data_sim["simulated [V]"], label="Simulated",
            linestyle="--", color="#0000FF")

    ax.set_xlabel("Current Density [mA/cm²]")
    ax.set_ylabel("Voltage [V]")
    ax.set_xlim(0, data_df["current dens. [mA/cm^2]"].max()*1.15)
    ax.set_ylim(data_df["voltage - measured [V]"].min()*0.95, data_df["voltage - measured [V]"].max()*1.1)
    ax.legend(loc='upper left')

    st.pyplot(fig)

st.write("---")
st.write("### General overview of parameters a, b, and c influence on overall shape of UI curve")
st.write("#### Description:")
st.write("Each plot shows three curves, in each column the curves relates to variation in a specific parameter")
st.write("**Parameter a:** presented in kiloOhm.cm^2, used values corresponds to 100, 250, and 1000 milliOhm.cm^2")
st.write("**Parameter b:** presented in Volt per decade, used values were 0.15, 0.20, and 0.25 V/dec")
st.write("**Parameter c:** presented in millAmper/cm^2, used values corresponds 0.1, 0.25, and 1.00 mA/cm^2")

st.image(
    "parameter_comparison.png",
    caption="visualization of changes caused be variation in parameters a, b, and c",
    use_container_width=False
)

st.write("#### Observation:")
st.write("""Varying the resistance influences the slope of curve, 
which is mainly visible and influencing the voltage at higher current""")
st.write("""Varying the Tafel slope shifts the whole curve vertically. 
Moreover, differences in curve shapes can be observed at lower current (lower Tafel slope causes "sharper" bending)""")
st.write("""Varying the Exchange current shifts the whole curve vertically.
Moreover, differences in curve shapes can be observed at lower current (lower value causes "sharper" bending)""")
st.write("""In summary, resistance influences mainly the slope and Tafel slope with Exchange current 
causes similar effects to overall shape of curve and its vertical position""")
