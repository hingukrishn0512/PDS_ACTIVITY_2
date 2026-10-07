from PDS_ACTIVITY_2.src.prediction_engine import predict_food_consumption


prediction = predict_food_consumption(
    expected_students=500,
    menu="Khichdi",
    day_of_week="Monday",
    weather="Hot",
    temperature_c=35,
    is_holiday=0,
    is_exam_day=0,
    special_event=0
)

print("Predicted food consumption:", prediction, "portions")