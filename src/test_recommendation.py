from PDS_ACTIVITY_2.src.recommendation_engine import generate_recommendation


result = generate_recommendation(
    expected_students=500,
    menu="Khichdi",
    day_of_week="Monday",
    weather="Hot",
    temperature_c=35,
    is_holiday=0,
    is_exam_day=0,
    special_event=0,
    safety_buffer_percent=5
)


print("\n===== FOOD PREPARATION RECOMMENDATION =====")

print(
    "Predicted consumption:",
    result["predicted_consumption"],
    "portions"
)

print(
    "Safety buffer:",
    result["safety_buffer"],
    "portions"
)

print(
    "Recommended preparation:",
    result["recommended_preparation"],
    "portions"
)