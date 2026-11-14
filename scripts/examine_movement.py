
import pandas as pd

path = "results/gravity_analysis/gravity_energy_timeseries.csv"
df = pd.read_csv(path)

print("COLUMNS:")
print(df.columns.tolist())

print("\nROW COUNT:", len(df))

print("\nSELECTED SIGNAL VALUES:")
for frame in [0, 56, 100, 133, 186, 219, 269, 270, 281]:
    row = df.loc[df["frame_index"] == frame]

    if not row.empty:
        print(
            f"Frame {frame}: "
            f"COM Y = {row.iloc[0]['com_y_m']:.6f}, "
            f"Time = {row.iloc[0]['time']:.6f}"
        )

print("\nGLOBAL EXTREMA:")
min_idx = df["com_y_m"].idxmin()
max_idx = df["com_y_m"].idxmax()

print("Minimum:")
print(df.loc[min_idx, ["frame_index", "time", "com_y_m"]])

print("Maximum:")
print(df.loc[max_idx, ["frame_index", "time", "com_y_m"]])
