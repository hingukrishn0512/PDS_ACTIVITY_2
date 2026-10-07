import pandas as pd
import streamlit as st

from PDS_ACTIVITY_2.src.prediction_engine import predict_food_consumption
from PDS_ACTIVITY_2.src.recommendation_engine import generate_recommendation
from PDS_ACTIVITY_2.src.simulator import run_simulation


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Food Waste Management System",
    page_icon="🍽️",
    layout="wide"
)


# ==========================================
# LOAD CLEANED DATASET
# ==========================================

DATA_PATH = "food_waste_management_dataset_cleaned.csv"

df = pd.read_csv(DATA_PATH)

# Convert date column
df["date"] = pd.to_datetime(df["date"])


# ==========================================
# HEADER
# ==========================================

st.title("🍽️ Food Waste Management System")

st.write(
    "AI-powered food demand prediction and "
    "waste reduction system."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Food Demand Prediction",
        "Before vs After Simulator"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Food Waste Analytics Dashboard")

    st.write(
        "Historical analysis of food preparation, "
        "consumption and waste."
    )

    # ==========================================
    # KEY METRICS
    # ==========================================

    total_records = len(df)

    avg_prepared = (
        df["food_prepared_portions"].mean()
    )

    avg_consumed = (
        df["food_consumed_portions"].mean()
    )

    avg_wasted = (
        df["food_wasted_portions"].mean()
    )

    avg_waste_percentage = (
        df["waste_percentage"].mean()
    )

    total_wasted = (
        df["food_wasted_portions"].sum()
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Records",
        total_records
    )

    col2.metric(
        "Avg Prepared",
        f"{avg_prepared:.1f}"
    )

    col3.metric(
        "Avg Consumed",
        f"{avg_consumed:.1f}"
    )

    col4.metric(
        "Avg Wasted",
        f"{avg_wasted:.1f}"
    )

    col5.metric(
        "Avg Waste",
        f"{avg_waste_percentage:.2f}%"
    )

    st.divider()

    # ==========================================
    # PREPARED VS CONSUMED VS WASTED
    # ==========================================

    st.subheader(
        "🍱 Food Preparation vs Consumption vs Waste"
    )

    summary_data = pd.DataFrame({
        "Category": [
            "Prepared",
            "Consumed",
            "Wasted"
        ],
        "Average Portions": [
            avg_prepared,
            avg_consumed,
            avg_wasted
        ]
    })

    st.bar_chart(
        summary_data.set_index("Category")
    )

    st.divider()

    # ==========================================
    # MENU WASTE ANALYSIS
    # ==========================================

    st.subheader(
        "🍛 Waste Percentage by Menu"
    )

    menu_analysis = (
        df.groupby("menu")["waste_percentage"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(menu_analysis)

    st.divider()

    # ==========================================
    # WEATHER ANALYSIS
    # ==========================================

    st.subheader(
        "🌦️ Waste Percentage by Weather"
    )

    weather_analysis = (
        df.groupby("weather")["waste_percentage"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(weather_analysis)

    st.divider()

    # ==========================================
    # DAY ANALYSIS
    # ==========================================

    st.subheader(
        "📅 Waste Percentage by Day"
    )

    day_analysis = (
        df.groupby("day_of_week")["waste_percentage"]
        .mean()
    )

    st.bar_chart(day_analysis)

    st.divider()

    # ==========================================
    # MONTHLY WASTE TREND
    # ==========================================

    st.subheader(
        "📈 Monthly Food Waste Trend"
    )

    monthly_waste = (
        df.groupby(
            df["date"].dt.to_period("M")
        )["food_wasted_portions"]
        .mean()
    )

    monthly_waste.index = (
        monthly_waste.index.astype(str)
    )

    st.line_chart(
        monthly_waste
    )

    # ==========================================
    # MONTHLY WASTE PERCENTAGE
    # ==========================================

    st.subheader(
        "📉 Monthly Waste Percentage"
    )

    monthly_waste_percentage = (
        df.groupby(
            df["date"].dt.to_period("M")
        )["waste_percentage"]
        .mean()
    )

    monthly_waste_percentage.index = (
        monthly_waste_percentage.index.astype(str)
    )

    st.line_chart(
        monthly_waste_percentage
    )

    st.divider()

    # ==========================================
    # SPECIAL CONDITION ANALYSIS
    # ==========================================

    st.subheader(
        "🎯 Special Condition Analysis"
    )

    c1, c2, c3 = st.columns(3)

    holiday_waste = (
        df.groupby("is_holiday")["waste_percentage"]
        .mean()
    )

    exam_waste = (
        df.groupby("is_exam_day")["waste_percentage"]
        .mean()
    )

    event_waste = (
        df.groupby("special_event")["waste_percentage"]
        .mean()
    )

    with c1:

        st.write("Holiday")

        holiday_display = (
            holiday_waste.rename(
                index={
                    0: "Normal Day",
                    1: "Holiday"
                }
            )
            .round(2)
        )

        st.dataframe(
            holiday_display,
            use_container_width=True
        )

    with c2:

        st.write("Exam Day")

        exam_display = (
            exam_waste.rename(
                index={
                    0: "Normal Day",
                    1: "Exam Day"
                }
            )
            .round(2)
        )

        st.dataframe(
            exam_display,
            use_container_width=True
        )

    with c3:

        st.write("Special Event")

        event_display = (
            event_waste.rename(
                index={
                    0: "Normal Day",
                    1: "Special Event"
                }
            )
            .round(2)
        )

        st.dataframe(
            event_display,
            use_container_width=True
        )

    st.divider()

    # ==========================================
    # MENU INTELLIGENCE
    # ==========================================

    st.subheader(
        "🍛 Menu Intelligence"
    )

    st.write(
        "Historical performance of each menu based on "
        "student demand, consumption and food waste."
    )

    menu_intelligence = (
        df.groupby("menu")
        .agg(
            avg_students=(
                "actual_students",
                "mean"
            ),
            avg_prepared=(
                "food_prepared_portions",
                "mean"
            ),
            avg_consumed=(
                "food_consumed_portions",
                "mean"
            ),
            avg_wasted=(
                "food_wasted_portions",
                "mean"
            ),
            avg_waste_percentage=(
                "waste_percentage",
                "mean"
            )
        )
        .sort_values(
            "avg_waste_percentage",
            ascending=False
        )
    )

    menu_intelligence = (
        menu_intelligence.round(2)
    )

    st.dataframe(
        menu_intelligence,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # TOTAL FOOD WASTE
    # ==========================================

    st.metric(
        "🌱 Total Food Waste",
        f"{total_wasted:,.0f} portions"
    )


# ============================================================
# FOOD DEMAND PREDICTION
# ============================================================

elif page == "Food Demand Prediction":

    st.header("🔮 Food Demand Prediction")

    st.write(
        "Enter the expected conditions to predict "
        "how many food portions are likely to be consumed."
    )

    st.divider()

    # ==========================================
    # INPUTS
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        expected_students = st.number_input(
            "Expected Students",
            min_value=1,
            max_value=2000,
            value=500,
            step=1
        )

        menu = st.selectbox(
            "Menu",
            [
                "Khichdi",
                "Poha",
                "Veg Pulao",
                "Rajma Rice",
                "Idli Sambar",
                "Dal Rice",
                "Dal Roti",
                "Chole Bhature",
                "Paneer Rice",
                "Pav Bhaji"
            ]
        )

        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        weather = st.selectbox(
            "Weather",
            [
                "Hot",
                "Cloudy",
                "Normal",
                "Cool",
                "Rainy"
            ]
        )

    with col2:

        temperature_c = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=30.0,
            step=0.5
        )

        is_holiday = st.selectbox(
            "Holiday?",
            ["No", "Yes"]
        )

        is_exam_day = st.selectbox(
            "Exam Day?",
            ["No", "Yes"]
        )

        special_event = st.selectbox(
            "Special Event?",
            ["No", "Yes"]
        )

    st.divider()

    # ==========================================
    # CONVERT YES / NO TO 0 / 1
    # ==========================================

    holiday_value = (
        1 if is_holiday == "Yes" else 0
    )

    exam_value = (
        1 if is_exam_day == "Yes" else 0
    )

    event_value = (
        1 if special_event == "Yes" else 0
    )

    # ==========================================
    # SAFETY BUFFER
    # ==========================================

    safety_buffer_percent = st.slider(
        "Safety Buffer (%)",
        min_value=0,
        max_value=20,
        value=5,
        step=1
    )

    # ==========================================
    # PREDICT
    # ==========================================

    if st.button(
        "🔮 Predict & Recommend",
        type="primary"
    ):

        prediction = predict_food_consumption(
            expected_students=expected_students,
            menu=menu,
            day_of_week=day_of_week,
            weather=weather,
            temperature_c=temperature_c,
            is_holiday=holiday_value,
            is_exam_day=exam_value,
            special_event=event_value
        )

        recommendation = generate_recommendation(
            expected_students=expected_students,
            menu=menu,
            day_of_week=day_of_week,
            weather=weather,
            temperature_c=temperature_c,
            is_holiday=holiday_value,
            is_exam_day=exam_value,
            special_event=event_value,
            safety_buffer_percent=safety_buffer_percent
        )

        st.success(
            "Prediction completed successfully."
        )

        st.divider()

        # ==========================================
        # RESULTS
        # ==========================================

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Predicted Consumption",
            f"{prediction} portions"
        )

        r2.metric(
            "Safety Buffer",
            f"{recommendation['safety_buffer']} portions"
        )

        r3.metric(
            "Recommended Preparation",
            f"{recommendation['recommended_preparation']} portions"
        )

        st.divider()

        st.subheader(
            "🍱 Preparation Recommendation"
        )

        st.info(
            f"For approximately **{expected_students} students**, "
            f"the model predicts consumption of approximately "
            f"**{prediction} portions**.\n\n"
            f"With a **{safety_buffer_percent}% safety buffer**, "
            f"the system recommends preparing approximately "
            f"**{recommendation['recommended_preparation']} portions**."
        )


# ============================================================
# BEFORE VS AFTER SIMULATOR
# ============================================================

elif page == "Before vs After Simulator":

    st.header(
        "🔄 Before vs After Simulator"
    )

    st.write(
        "Compare traditional food preparation "
        "with the AI-assisted recommendation."
    )

    st.divider()

    # ==========================================
    # INPUTS
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        expected_students = st.number_input(
            "Expected Students",
            min_value=1,
            max_value=2000,
            value=500,
            step=1,
            key="sim_students"
        )

        menu = st.selectbox(
            "Menu",
            [
                "Khichdi",
                "Poha",
                "Veg Pulao",
                "Rajma Rice",
                "Idli Sambar",
                "Dal Rice",
                "Dal Roti",
                "Chole Bhature",
                "Paneer Rice",
                "Pav Bhaji"
            ],
            key="sim_menu"
        )

        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ],
            key="sim_day"
        )

        weather = st.selectbox(
            "Weather",
            [
                "Hot",
                "Cloudy",
                "Normal",
                "Cool",
                "Rainy"
            ],
            key="sim_weather"
        )

    with col2:

        temperature_c = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=30.0,
            step=0.5,
            key="sim_temperature"
        )

        is_holiday = st.selectbox(
            "Holiday?",
            ["No", "Yes"],
            key="sim_holiday"
        )

        is_exam_day = st.selectbox(
            "Exam Day?",
            ["No", "Yes"],
            key="sim_exam"
        )

        special_event = st.selectbox(
            "Special Event?",
            ["No", "Yes"],
            key="sim_event"
        )

    food_cost_per_portion = st.number_input(
        "Food Cost per Portion (₹)",
        min_value=1.0,
        max_value=500.0,
        value=30.0,
        step=1.0
    )

    safety_buffer_percent = st.slider(
        "Safety Buffer (%)",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        key="sim_buffer"
    )

    # ==========================================
    # CONVERT YES / NO
    # ==========================================

    holiday_value = (
        1 if is_holiday == "Yes" else 0
    )

    exam_value = (
        1 if is_exam_day == "Yes" else 0
    )

    event_value = (
        1 if special_event == "Yes" else 0
    )

    st.divider()

    # ==========================================
    # RUN SIMULATION
    # ==========================================

    if st.button(
        "🔄 Run Before vs After Simulation",
        type="primary"
    ):

        result = run_simulation(
            expected_students=expected_students,
            menu=menu,
            day_of_week=day_of_week,
            weather=weather,
            temperature_c=temperature_c,
            is_holiday=holiday_value,
            is_exam_day=exam_value,
            special_event=event_value,
            food_cost_per_portion=food_cost_per_portion,
            safety_buffer_percent=safety_buffer_percent
        )

        st.success(
            "Simulation completed successfully."
        )

        # ======================================
        # TRADITIONAL APPROACH
        # ======================================

        st.subheader(
            "📦 Traditional Approach"
        )

        b1, b2, b3 = st.columns(3)

        b1.metric(
            "Food Prepared",
            f"{result['traditional_preparation']} portions"
        )

        b2.metric(
            "Estimated Waste",
            f"{result['traditional_waste']} portions"
        )

        b3.metric(
            "Estimated Waste",
            f"{result['traditional_waste'] * 0.35:.2f} kg"
        )

        st.divider()

        # ======================================
        # AI APPROACH
        # ======================================

        st.subheader(
            "🤖 AI-Assisted Approach"
        )

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Predicted Consumption",
            f"{result['predicted_consumption']} portions"
        )

        a2.metric(
            "Recommended Preparation",
            f"{result['ai_preparation']} portions"
        )

        a3.metric(
            "Estimated Waste",
            f"{result['ai_waste']} portions"
        )

        st.divider()

        # ======================================
        # IMPACT
        # ======================================

        st.subheader(
            "🌱 Potential Impact"
        )

        i1, i2, i3 = st.columns(3)

        i1.metric(
            "Portions Potentially Saved",
            f"{result['portions_saved']}"
        )

        i2.metric(
            "Food Potentially Saved",
            f"{result['food_saved_kg']} kg"
        )

        i3.metric(
            "Potential Cost Saved",
            f"₹{result['cost_saved_inr']:,.0f}"
        )

        st.divider()

        st.metric(
            "Estimated Waste Reduction",
            f"{result['waste_reduction_percentage']}%"
        )

        st.caption(
            "⚠️ These are model-based simulation estimates. "
            "Actual savings may differ in real-world operation."
        )