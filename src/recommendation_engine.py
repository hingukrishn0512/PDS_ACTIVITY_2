from PDS_ACTIVITY_2.src.prediction_engine import predict_food_consumption


def generate_recommendation(
    expected_students,
    menu,
    day_of_week,
    weather,
    temperature_c,
    is_holiday,
    is_exam_day,
    special_event,
    safety_buffer_percent=5
):

    # ----------------------------------
    # 1. Predict consumption
    # ----------------------------------

    predicted_consumption = predict_food_consumption(
        expected_students=expected_students,
        menu=menu,
        day_of_week=day_of_week,
        weather=weather,
        temperature_c=temperature_c,
        is_holiday=is_holiday,
        is_exam_day=is_exam_day,
        special_event=special_event
    )

    # ----------------------------------
    # 2. Calculate safety buffer
    # ----------------------------------

    safety_buffer = round(
        predicted_consumption
        * safety_buffer_percent
        / 100
    )

    # ----------------------------------
    # 3. Recommended preparation
    # ----------------------------------

    recommended_preparation = (
        predicted_consumption
        + safety_buffer
    )

    return {
        "predicted_consumption": predicted_consumption,
        "safety_buffer": safety_buffer,
        "recommended_preparation": recommended_preparation
    }