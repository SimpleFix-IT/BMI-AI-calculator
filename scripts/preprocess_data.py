# import pandas as pd

# # ✅ Apne dataset ka sahi path yahan update karein
# DATASET_PATH = "/home/nsdev/Downloads/archive/height_weight.csv"

# image_path = sys.argv[1]
# dataset_path = sys.argv[2] 
# # 📌 1️⃣ Load Dataset
# df = pd.read_csv(DATASET_PATH)
import sys
import pandas as pd

image_path = sys.argv[1]
dataset_path = sys.argv[2]  # ✅ Laravel se correct dataset path aa raha hai

# 📌 1️⃣ Load Dataset
df = pd.read_csv(dataset_path)  # ✅ Ab hardcoded path ki zaroorat nahi

# 📌 2️⃣ Convert Height from Inches to Centimeters (1 inch = 2.54 cm)
df["Height (cm)"] = df["Height(Inches)"] * 2.54

# 📌 3️⃣ Convert Weight from Pounds to Kilograms (1 pound = 0.453592 kg)
df["Weight (kg)"] = df["Weight(Pounds)"] * 0.453592

# 📌 4️⃣ Drop Old Columns
df = df[["Height (cm)", "Weight (kg)"]]

# 📌 5️⃣ Save Cleaned Data

print("\n✅ Data preprocessing complete! Cleaned dataset saved at")
