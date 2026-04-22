import pandas as pd
import glob
import math

files = glob.glob("dataset/*.csv")

cleaned_data = []

for file in files:
    df = pd.read_csv(file)

    print(f"Processing {file}...")

    for index, row in df.iterrows():
        values = row.values.tolist()

        # Last column = label
        label = values[-1]
        coords = values[:-1]

        # 1. Remove bad rows (must be exactly 63 values)
        if len(coords) != 63:
            continue

        # Convert to float
        coords = list(map(float, coords))

        # 2. Get wrist (first point)
        base_x = coords[0]
        base_y = coords[1]
        base_z = coords[2]

        # 3. Normalize relative to wrist
        normalized = []
        for i in range(0, 63, 3):
            normalized.append(coords[i] - base_x)
            normalized.append(coords[i+1] - base_y)
            normalized.append(coords[i+2] - base_z)

        # 4. Scale normalization
        # distance wrist → middle finger tip (point 12)
        mx = coords[12*3]
        my = coords[12*3 + 1]
        mz = coords[12*3 + 2]

        scale = math.sqrt((mx-base_x)**2 + (my-base_y)**2 + (mz-base_z)**2)

        if scale != 0:
            normalized = [v / scale for v in normalized]

        # Add label back
        normalized.append(label)

        cleaned_data.append(normalized)

# 5. Create final dataframe
final_df = pd.DataFrame(cleaned_data)

# Save
final_df.to_csv("final_dataset.csv", index=False)

print("Cleaned & merged dataset saved as final_dataset.csv")