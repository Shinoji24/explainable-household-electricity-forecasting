import streamlit as st
import pandas as pd
import random
import pickle
import shap
import matplotlib.pyplot as plt
import textwrap


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="WattScope",
    layout="wide"
)


# ==========================================
# LOAD SAVED MODELS
# ==========================================

@st.cache_resource
def load_forecasting_model():

    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


best_model = load_forecasting_model()


@st.cache_resource
def load_anomaly_model():

    with open("isolation_forest.pkl", "rb") as file:
        model = pickle.load(file)

    return model


isolation_model = load_anomaly_model()


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_consumption(input_values):

    input_df = pd.DataFrame(
        [input_values],
        columns=["lag 1", "lag 2", "lag 3", "lag_24"]
    )

    prediction = best_model.predict(input_df)

    return prediction[0]


# ==========================================
# FUTURE FORECASTING FUNCTION
# ==========================================

def generate_future_forecast(hourly_df, hours=24):

    history = hourly_df["Global_active_power"].dropna().tolist()

    if len(history) < 24:
        raise ValueError(
            "At least 24 valid hourly readings are required."
        )

    recent_data = hourly_df["Global_active_power"].tail(24)

    if len(recent_data) < 24 or recent_data.isna().any():
        raise ValueError(
            "The latest 24 hours contain missing readings. "
            "Future forecasting cannot continue."
        )

    future_predictions = []

    for step in range(hours):

        input_values = {
            "lag 1": history[-1],
            "lag 2": history[-2],
            "lag 3": history[-3],
            "lag_24": history[-24]
        }

        prediction = predict_consumption(input_values)

        future_predictions.append(prediction)

        history.append(prediction)

    last_datetime = hourly_df.index[-1]

    future_dates = pd.date_range(
        start=last_datetime + pd.Timedelta(hours=1),
        periods=hours,
        freq="h"
    )

    future_df = pd.DataFrame({
        "Datetime": future_dates,
        "Predicted_Global_active_power": future_predictions
    })

    return future_df


# ==========================================
# HTML RENDERING FUNCTION
# ==========================================

def render_html(content):

    clean_content = textwrap.dedent(content).strip()

    st.html(clean_content)


# ==========================================
# LOAD CUSTOM CSS
# ==========================================

with open("style.css", "r") as file:
    css = file.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# ==========================================
# INITIALIZE PAGE
# ==========================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.title("⚡ WattScope")

    st.caption("Intelligent Electricity Analytics")

    st.divider()

    if st.button("Overview", use_container_width=True):
        st.session_state.page = "Overview"

    if st.button("Consumption Forecasting", use_container_width=True):
        st.session_state.page = "Consumption Forecasting"

    if st.button("Anomaly Detection", use_container_width=True):
        st.session_state.page = "Anomaly Detection"

    if st.button("Explainable AI", use_container_width=True):
        st.session_state.page = "Explainable AI"


# ==========================================
# OVERVIEW PAGE
# ==========================================

if st.session_state.page == "Overview":

    render_html("""
        <div class="hero-section">

            <div class="hero-label">
                HOUSEHOLD ENERGY INTELLIGENCE
            </div>

            <h1 class="hero-title">
                Smarter Insights.<br>
                Better Energy Decisions.
            </h1>

            <p class="hero-description">
                Monitor electricity consumption, forecast future
                usage and identify unusual patterns with
                Machine Learning and Explainable AI.
            </p>

            <div class="hero-tags">
                <span>ENERGY FORECASTING</span>
                <span>ANOMALY DETECTION</span>
                <span>EXPLAINABLE AI</span>
            </div>

            <div class="hero-watermark">⚡</div>

        </div>
    """)

    render_html("""
        <div class="section-heading">
            <h2>Explore WattScope</h2>
            <p>
                Intelligent tools to understand your household energy usage.
            </p>
        </div>
    """)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:

        render_html("""
            <div class="overview-card">
                <div class="card-icon">↗</div>
                <div class="card-number">01 / FORECAST</div>
                <h3>Consumption Forecasting</h3>
                <p>
                    Predict upcoming household electricity
                    consumption using historical readings
                    and an XGBoost model.
                </p>
                <div class="card-bottom">ENERGY PREDICTION</div>
            </div>
        """)

    with col2:

        render_html("""
            <div class="overview-card">
                <div class="card-icon">⌁</div>
                <div class="card-number">02 / DETECTION</div>
                <h3>Anomaly Detection</h3>
                <p>
                    Identify unusual electricity usage
                    patterns using the Isolation Forest
                    machine learning algorithm.
                </p>
                <div class="card-bottom">UNUSUAL USAGE ANALYSIS</div>
            </div>
        """)

    with col3:

        render_html("""
            <div class="overview-card">
                <div class="card-icon">✳</div>
                <div class="card-number">03 / EXPLANATION</div>
                <h3>Explainable AI</h3>
                <p>
                    Understand how previous electricity
                    readings influence predictions through
                    SHAP explanations.
                </p>
                <div class="card-bottom">MODEL TRANSPARENCY</div>
            </div>
        """)

    render_html("""
        <div class="overview-footer">
            <span>WATTSCOPE</span>
            <span>INTELLIGENT HOUSEHOLD ELECTRICITY ANALYTICS</span>
        </div>
    """)


# ==========================================
# CONSUMPTION FORECASTING PAGE
# ==========================================

elif st.session_state.page == "Consumption Forecasting":

    st.title("Consumption Forecasting")

    st.markdown(
        "Forecast household electricity consumption "
        "using historical readings or manual inputs."
    )

    st.divider()

    st.subheader("Start Your Analysis")

    st.write(
        "Choose how you want to provide your electricity consumption data."
    )

    if "input_method" not in st.session_state:
        st.session_state.input_method = "CSV Upload"

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button("📁 CSV Upload", use_container_width=True):
            st.session_state.input_method = "CSV Upload"

    with col2:

        if st.button("✍️ Manual Input", use_container_width=True):
            st.session_state.input_method = "Manual Input"

    with col3:

        if st.button("⚡ Demo Data", use_container_width=True):
            st.session_state.input_method = "Demo Data"

    st.divider()

    # CSV UPLOAD INTERFACE

    if st.session_state.input_method == "CSV Upload":

        st.markdown("#### Upload Your Electricity Data")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload your historical electricity consumption dataset."
        )

        if uploaded_file is not None:

            try:

                csv_data = pd.read_csv(
                    uploaded_file,
                    na_values=["?"]
                )

                st.success("CSV file uploaded successfully!")

                st.write("Number of rows:", csv_data.shape[0])

                st.write("Number of columns:", csv_data.shape[1])

                st.write("Data Preview")

                st.dataframe(
                    csv_data.head(10),
                    use_container_width=True
                )

                has_power = (
                    "Global_active_power" in csv_data.columns
                )

                has_datetime = (
                    "Datetime" in csv_data.columns
                    or {"Date", "Time"}.issubset(csv_data.columns)
                )

                if has_power and has_datetime:

                    if "Datetime" not in csv_data.columns:

                        csv_data["Datetime"] = pd.to_datetime(
                            csv_data["Date"].astype(str) + " " +
                            csv_data["Time"].astype(str),
                            dayfirst=True,
                            errors="coerce"
                        )

                    else:

                        csv_data["Datetime"] = pd.to_datetime(
                            csv_data["Datetime"],
                            errors="coerce"
                        )

                    csv_data = csv_data.dropna(
                        subset=["Datetime"]
                    )

                    csv_data = csv_data.sort_values(
                        "Datetime"
                    ).reset_index(drop=True)

                    target = "Global_active_power"

                    csv_data[target] = pd.to_numeric(
                        csv_data[target],
                        errors="coerce"
                    )

                    csv_data = csv_data.set_index("Datetime")

                    missing = csv_data[target].isna()

                    missing_groups = missing.ne(
                        missing.shift()
                    ).cumsum()

                    gap_info = (
                        pd.DataFrame({
                            "group": missing_groups[missing],
                            "Datetime": csv_data.index[missing]
                        })
                        .groupby("group")["Datetime"]
                        .agg(["min", "max"])
                    )

                    gap_duration = (
                        (gap_info["max"] - gap_info["min"])
                        .dt.total_seconds() / 60
                    ) + 1

                    short_gap_groups = gap_duration[
                        gap_duration <= 5
                    ].index

                    short_gap_mask = (
                        missing &
                        missing_groups.isin(short_gap_groups)
                    )

                    interpolated = csv_data[target].interpolate(
                        method="time"
                    )

                    csv_data.loc[
                        short_gap_mask, target
                    ] = interpolated.loc[short_gap_mask]

                    hourly_df = (
                        csv_data[[target]]
                        .resample("h")
                        .mean()
                    )

                    forecast_df = hourly_df[[target]].copy()

                    forecast_df["lag 1"] = (
                        forecast_df[target].shift(1)
                    )

                    forecast_df["lag 2"] = (
                        forecast_df[target].shift(2)
                    )

                    forecast_df["lag 3"] = (
                        forecast_df[target].shift(3)
                    )

                    forecast_df["lag_24"] = (
                        forecast_df[target].shift(24)
                    )

                    forecast_df = forecast_df.dropna().copy()

                    st.session_state.hourly_df = hourly_df

                    forecast_df = forecast_df.reset_index()

                    st.session_state.forecast_df = forecast_df

                    st.success(
                        "Hourly CSV preprocessing completed!"
                    )

                    st.write(
                        "Prepared hourly forecasting records:",
                        forecast_df.shape[0]
                    )

                    st.write("Generated Hourly Lag Features")

                    st.dataframe(
                        forecast_df[
                            [
                                "Datetime",
                                "Global_active_power",
                                "lag 1",
                                "lag 2",
                                "lag 3",
                                "lag_24"
                            ]
                        ].head(10),
                        use_container_width=True
                    )

                    # HISTORICAL PREDICTIONS

                    if st.button("Generate CSV Predictions"):

                        csv_results = forecast_df.copy()

                        X_csv = csv_results[
                            ["lag 1", "lag 2", "lag 3", "lag_24"]
                        ]

                        csv_results[
                            "Predicted_Global_active_power"
                        ] = best_model.predict(X_csv)

                        st.session_state.csv_forecast_results = csv_results

                        st.success(
                            "Historical predictions generated successfully!"
                        )

                        st.subheader("Historical Prediction Results")

                        st.dataframe(
                            csv_results[
                                [
                                    "Datetime",
                                    "Global_active_power",
                                    "Predicted_Global_active_power"
                                ]
                            ].head(10),
                            use_container_width=True
                        )

                        st.subheader("Actual vs Predicted Consumption")

                        chart_data = (
                            csv_results
                            .set_index("Datetime")[
                                [
                                    "Global_active_power",
                                    "Predicted_Global_active_power"
                                ]
                            ]
                            .head(500)
                        )

                        st.line_chart(chart_data)

                    # FUTURE FORECASTING

                    st.divider()

                    st.subheader(
                        "Future Electricity Consumption Forecast"
                    )

                    if st.button("Generate 24-Hour Future Forecast"):

                        try:

                            future_df = generate_future_forecast(
                                st.session_state.hourly_df,
                                hours=24
                            )

                            st.session_state.future_forecast_df = future_df

                            st.success(
                                "24-hour future forecast generated successfully!"
                            )

                            st.dataframe(
                                future_df,
                                use_container_width=True
                            )

                            st.subheader("Future Consumption Trend")

                            future_chart = future_df.set_index("Datetime")

                            st.line_chart(
                                future_chart[
                                    ["Predicted_Global_active_power"]
                                ]
                            )

                        except Exception as e:

                            st.error(
                                f"Unable to generate future forecast: {e}"
                            )

                else:

                    st.warning(
                        "CSV must contain Global_active_power "
                        "and either Datetime or both Date and Time columns."
                    )

            except Exception as e:

                st.error(f"Unable to read CSV file: {e}")

    # MANUAL INPUT INTERFACE

    elif st.session_state.input_method == "Manual Input":

        st.markdown("#### Enter Previous Electricity Readings")

        st.write(
            "Enter the previous hourly electricity consumption readings "
            "to predict the next reading."
        )

        with st.form("manual_forecast_form"):

            col1, col2 = st.columns(2)

            with col1:

                lag_1 = st.number_input(
                    "Previous Reading 1 (kW)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )

                lag_2 = st.number_input(
                    "Previous Reading 2 (kW)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )

            with col2:

                lag_3 = st.number_input(
                    "Previous Reading 3 (kW)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )

                lag_24 = st.number_input(
                    "Reading 24 Hours Ago (kW)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )

            manual_submitted = st.form_submit_button(
                "Predict",
                use_container_width=True
            )

        if manual_submitted:

            manual_values = {
                "lag 1": lag_1,
                "lag 2": lag_2,
                "lag 3": lag_3,
                "lag_24": lag_24
            }

            prediction = predict_consumption(manual_values)

            st.success("Forecast generated successfully!")

            st.metric(
                "Predicted Electricity Consumption",
                f"{prediction:.3f} kW"
            )

    # DEMO DATA INTERFACE

    elif st.session_state.input_method == "Demo Data":

        st.markdown("#### Explore WattScope with Demo Data")

        st.write(
            "Generate sample electricity readings "
            "and use them to test the forecasting model."
        )

        if st.button("Generate Demo Values"):

            demo_values = {
                "lag 1": round(random.uniform(0.5, 3.0), 2),
                "lag 2": round(random.uniform(0.5, 3.0), 2),
                "lag 3": round(random.uniform(0.5, 3.0), 2),
                "lag_24": round(random.uniform(0.5, 3.0), 2)
            }

            st.session_state.demo_forecast_input = demo_values

        if "demo_forecast_input" in st.session_state:

            demo_values = st.session_state.demo_forecast_input

            st.write("Generated Electricity Readings")

            col1, col2 = st.columns(2)

            with col1:

                st.number_input(
                    "Previous Reading 1 (kW)",
                    value=demo_values["lag 1"],
                    disabled=True,
                    key="demo_lag1"
                )

                st.number_input(
                    "Previous Reading 2 (kW)",
                    value=demo_values["lag 2"],
                    disabled=True,
                    key="demo_lag2"
                )

            with col2:

                st.number_input(
                    "Previous Reading 3 (kW)",
                    value=demo_values["lag 3"],
                    disabled=True,
                    key="demo_lag3"
                )

                st.number_input(
                    "Reading 24 Hours Ago (kW)",
                    value=demo_values["lag_24"],
                    disabled=True,
                    key="demo_lag24"
                )

            if st.button("Predict", key="demo_predict"):

                prediction = predict_consumption(demo_values)

                st.success("Forecast generated successfully!")

                st.metric(
                    "Predicted Electricity Consumption",
                    f"{prediction:.3f} kW"
                )


# ==========================================
# ANOMALY DETECTION PAGE
# ==========================================

elif st.session_state.page == "Anomaly Detection":

    st.title("Anomaly Detection")

    st.markdown(
        "Identify unusual household electricity consumption patterns."
    )

    st.divider()

    if "hourly_df" in st.session_state:

        anomaly_df = (
            st.session_state.hourly_df
            .dropna(subset=["Global_active_power"])
            .copy()
        )

        if not anomaly_df.empty:

            anomaly_input = anomaly_df[
                ["Global_active_power"]
            ]

            anomaly_df["Anomaly"] = isolation_model.predict(
                anomaly_input
            )

            anomaly_df["Status"] = anomaly_df["Anomaly"].map({
                1: "Normal",
                -1: "Anomaly"
            })

            anomaly_df = anomaly_df.reset_index()

            total_readings = len(anomaly_df)

            normal_readings = (
                anomaly_df["Anomaly"] == 1
            ).sum()

            unusual_readings = (
                anomaly_df["Anomaly"] == -1
            ).sum()

            anomaly_percentage = (
                unusual_readings / total_readings
            ) * 100 if total_readings > 0 else 0

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Readings", total_readings)

            col2.metric("Normal Readings", normal_readings)

            col3.metric("Unusual Readings", unusual_readings)

            st.metric(
                "Anomaly Percentage",
                f"{anomaly_percentage:.2f}%"
            )

            st.divider()

            st.subheader("Anomaly Detection Results")

            anomaly_records = anomaly_df[
                anomaly_df["Anomaly"] == -1
            ]

            st.dataframe(
                anomaly_records[
                    ["Datetime", "Global_active_power", "Status"]
                ],
                use_container_width=True
            )

            st.subheader("Electricity Consumption Pattern")

            chart_df = anomaly_df.set_index("Datetime")

            st.line_chart(
                chart_df[["Global_active_power"]].head(500)
            )

        else:

            st.warning(
                "No valid hourly electricity readings were found."
            )

    else:

        st.warning(
            "Please upload your electricity CSV file "
            "in the Consumption Forecasting page first."
        )


# ==========================================
# EXPLAINABLE AI PAGE
# ==========================================

elif st.session_state.page == "Explainable AI":

    st.title("Explainable AI")

    st.markdown(
        "Understand how previous electricity readings "
        "influence the XGBoost forecast."
    )

    st.divider()

    if "forecast_df" in st.session_state:

        explain_df = st.session_state.forecast_df.copy()

        if not explain_df.empty:

            feature_columns = [
                "lag 1",
                "lag 2",
                "lag 3",
                "lag_24"
            ]

            X_explain = explain_df[feature_columns]

            st.subheader("Select a Historical Reading")

            selected_index = st.slider(
                "Choose a record",
                min_value=0,
                max_value=len(X_explain) - 1,
                value=0
            )

            selected_input = X_explain.iloc[[selected_index]]

            actual_value = explain_df.iloc[selected_index][
                "Global_active_power"
            ]

            predicted_value = best_model.predict(selected_input)[0]

            col1, col2 = st.columns(2)

            col1.metric(
                "Actual Consumption",
                f"{actual_value:.3f} kW"
            )

            col2.metric(
                "Predicted Consumption",
                f"{predicted_value:.3f} kW"
            )

            st.divider()

            st.subheader("SHAP Prediction Explanation")

            try:

                explainer = shap.TreeExplainer(best_model)

                shap_values = explainer(selected_input)

                shap.plots.waterfall(
                    shap_values[0],
                    show=False
                )

                st.pyplot(plt.gcf())

                plt.close()

                st.caption(
                    "The waterfall plot shows how each lag feature "
                    "contributes to the model's prediction."
                )

            except Exception as e:

                st.error(
                    f"Unable to generate SHAP explanation: {e}"
                )

        else:

            st.warning(
                "No forecasting records are available for explanation."
            )

    else:

        st.warning(
            "Please upload and preprocess your electricity CSV "
            "in the Consumption Forecasting page first."
        )

