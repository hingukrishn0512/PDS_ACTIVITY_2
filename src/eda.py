import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv(
    "food_waste_management_dataset_cleaned.csv"
)


#  Prepared vs Consumed vs Wasted


# overall = [
#     df["food_prepared_portions"].mean(),
#     df["food_consumed_portions"].mean(),
#     df["food_wasted_portions"].mean()
# ]

# labels = ["Prepared", "Consumed", "Wasted"]

# plt.figure(figsize=(8, 5))
# sns.barplot(x=labels, y=overall)

# plt.title("Average Food Prepared, Consumed and Wasted")
# plt.ylabel("Average Portions")
# plt.xlabel("")

# plt.tight_layout()
# plt.show()


# # -----------------------------
# # 2. Waste Percentage by Menu
# # -----------------------------

# menu_waste = (
#     df.groupby("menu")["waste_percentage"]
#     .mean()
#     .sort_values(ascending=False)
# )

# plt.figure(figsize=(10, 6))
# sns.barplot(
#     x=menu_waste.values,
#     y=menu_waste.index
# )

# plt.title("Average Food Waste Percentage by Menu")
# plt.xlabel("Waste Percentage (%)")
# plt.ylabel("Menu")

# plt.tight_layout()
# plt.show()


# # -----------------------------
# # 3. Waste Percentage by Day
# # -----------------------------

# day_order = [
#     "Monday",
#     "Tuesday",
#     "Wednesday",
#     "Thursday",
#     "Friday",
#     "Saturday",
#     "Sunday"
# ]

# day_waste = (
#     df.groupby("day_of_week")["waste_percentage"]
#     .mean()
#     .reindex(day_order)
# )

# plt.figure(figsize=(9, 5))
# sns.barplot(
#     x=day_waste.index,
#     y=day_waste.values
# )

# plt.title("Average Food Waste Percentage by Day")
# plt.xlabel("Day")
# plt.ylabel("Waste Percentage (%)")

# plt.xticks(rotation=30)

# plt.tight_layout()
# plt.show()

#   CORRELATION ANALYSIS

correlation_columns = [
    "actual_students",
    "food_prepared_portions",
    "food_consumed_portions",
    "food_wasted_portions",
    "temperature_c",
    "waste_percentage"
]
correlation = df[correlation_columns].corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="turbo", vmin=-1, vmax=1)
# 3. Add column labels to x and y axes
plt.xticks(range(len(correlation_columns)), correlation_columns, rotation=45, ha='right')
plt.yticks(range(len(correlation_columns)), correlation_columns)

plt.tight_layout()
plt.show()

# -----------------------------
# 5. Attendance vs Food Waste
# -----------------------------

# plt.figure(figsize=(9, 6))

# sns.scatterplot(
#     data=df,
#     x="actual_students",
#     y="food_wasted_portions",
#     hue="menu",
#     alpha=0.7
# )

# plt.title("Actual Students vs Food Waste")
# plt.xlabel("Actual Students")
# plt.ylabel("Food Wasted (Portions)")

# plt.tight_layout()
# plt.show()


# # -----------------------------
# # 6. Prepared Food vs Waste
# # -----------------------------

# plt.figure(figsize=(9, 6))

# sns.scatterplot(
#     data=df,
#     x="food_prepared_portions",
#     y="food_wasted_portions",
#     hue="menu",
#     alpha=0.7
# )

# plt.title("Food Prepared vs Food Wasted")
# plt.xlabel("Food Prepared (Portions)")
# plt.ylabel("Food Wasted (Portions)")

# plt.tight_layout()
# plt.show()

# print("\n===== HOLIDAY ANALYSIS =====")

# print(
#     df.groupby("is_holiday")["waste_percentage"]
#     .mean()
#     .round(2)
# )

# print("\n===== EXAM DAY ANALYSIS =====")

# print(
#     df.groupby("is_exam_day")["waste_percentage"]
#     .mean()
#     .round(2)
# )

# print("\n===== SPECIAL EVENT ANALYSIS =====")

# print(
#     df.groupby("special_event")["waste_percentage"]
#     .mean()
#     .round(2)
# )