import numpy as np
import pandas as pd

df_seg = pd.read_csv("ED_results.csv")
df_edge = pd.read_csv("SEG_results.csv")

df = pd.merge(df_seg, df_edge, on="tileId", suffixes=("_seg", "_edge"))

df["gtArea"] = df["gtArea_seg"]

df["err_area_seg"] = (df["extArea_seg"] - df["gtArea"]).abs() / df["gtArea"]

df["err_area_edge"] = (df["extArea_edge"] - df["gtArea"]).abs() / df["gtArea"]

df["err_area_bbox"] = (df["bboxArea"] - df["gtArea"]).abs() / df["gtArea"]


print(f"Modello Segmentazione: {df['err_area_seg'].mean() * 100:.2f}%")
print(f"Edge Detection: {df['err_area_edge'].mean() * 100:.2f}%")
print(f"Stima Bounding Box:    {df['err_area_bbox'].mean() * 100:.2f}%")
