from PDS_ACTIVITY_2.src.simulator import run_simulation


result = run_simulation(
    expected_students=500,
    menu="Khichdi",
    day_of_week="Monday",
    weather="Hot",
    temperature_c=35,
    is_holiday=0,
    is_exam_day=0,
    special_event=0,
    food_cost_per_portion=30,
    safety_buffer_percent=5
)


print("\n===== BEFORE vs AFTER SIMULATION =====")

print(
    "Expected students:",
    result["expected_students"]
)

print(
    "\n--- Traditional Approach ---"
)

print(
    "Food prepared:",
    result["traditional_preparation"],
    "portions"
)

print(
    "Estimated waste:",
    result["traditional_waste"],
    "portions"
)

print(
    "\n--- AI-Assisted Approach ---"
)

print(
    "Predicted consumption:",
    result["predicted_consumption"],
    "portions"
)

print(
    "Recommended preparation:",
    result["ai_preparation"],
    "portions"
)

print(
    "Estimated waste:",
    result["ai_waste"],
    "portions"
)

print(
    "\n--- Impact ---"
)

print(
    "Portions potentially saved:",
    result["portions_saved"]
)

print(
    "Estimated waste reduction:",
    result["waste_reduction_percentage"],
    "%"
)
print(
    "Food potentially saved:",
    result["food_saved_kg"],
    "kg"
)

print(
    "Potential food cost saved: ₹",
    result["cost_saved_inr"]
)