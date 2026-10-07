from PDS_ACTIVITY_2.src.recommendation_engine import generate_recommendation


# Average weight of one food portion
KG_PER_PORTION = 0.35


def run_simulation(
    expected_students,
    menu,
    day_of_week,
    weather,
    temperature_c,
    is_holiday,
    is_exam_day,
    special_event,
    food_cost_per_portion,
    safety_buffer_percent=5
):

    # ==========================================
    # AI RECOMMENDATION
    # ==========================================

    recommendation = generate_recommendation(
        expected_students=expected_students,
        menu=menu,
        day_of_week=day_of_week,
        weather=weather,
        temperature_c=temperature_c,
        is_holiday=is_holiday,
        is_exam_day=is_exam_day,
        special_event=special_event,
        safety_buffer_percent=safety_buffer_percent
    )

    predicted_consumption = recommendation[
        "predicted_consumption"
    ]

    recommended_preparation = recommendation[
        "recommended_preparation"
    ]

    # ==========================================
    # TRADITIONAL APPROACH
    # ==========================================

    traditional_preparation = expected_students

    traditional_waste = max(
        0,
        traditional_preparation - predicted_consumption
    )

    # ==========================================
    # AI APPROACH
    # ==========================================

    ai_waste = max(
        0,
        recommended_preparation - predicted_consumption
    )

    # ==========================================
    # IMPACT
    # ==========================================

    portions_saved = max(
        0,
        traditional_waste - ai_waste
    )

    waste_reduction_percentage = (
        portions_saved / traditional_waste * 100
        if traditional_waste > 0
        else 0
    )

    # ==========================================
    # FOOD WEIGHT SAVED
    # ==========================================

    food_saved_kg = (
        portions_saved * KG_PER_PORTION
    )

    # ==========================================
    # COST SAVED
    # ==========================================

    cost_saved_inr = (
        portions_saved * food_cost_per_portion
    )

    return {
        "expected_students": expected_students,

        "predicted_consumption":
            predicted_consumption,

        "traditional_preparation":
            traditional_preparation,

        "traditional_waste":
            traditional_waste,

        "ai_preparation":
            recommended_preparation,

        "ai_waste":
            ai_waste,

        "portions_saved":
            portions_saved,

        "waste_reduction_percentage":
            round(
                waste_reduction_percentage,
                2
            ),

        "food_saved_kg":
            round(
                food_saved_kg,
                2
            ),

        "cost_saved_inr":
            round(
                cost_saved_inr,
                2
            )
    }