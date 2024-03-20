import pandas as pd
import matplotlib.pyplot as plt  # Importar librería para graficar
import numpy as np
from scipy.interpolate import CubicSpline

#read CSV with pandas
ground_truth_df = pd.read_csv("mnyo_ground_truth.csv", sep=" ", header=None, names=["x1", "x2"])

mediciones_df = pd.read_csv("mnyo_mediciones.csv", sep=" ", header=None, names=["x1", "x2"])


#interpolate the trajectory based on index
interpolated_trajectory_x1 = CubicSpline(mediciones_df.index, mediciones_df["x1"])
interpolated_trajectory_x2 = CubicSpline(mediciones_df.index, mediciones_df["x2"])


#generate 1000 points to graph
points_to_graph = np.linspace(mediciones_df.index.min(), mediciones_df.index.max(), 1000)


#plot init
plt.figure(figsize=(12, 5))

#plot the ground truth
plt.plot(ground_truth_df["x1"], ground_truth_df["x2"], label='Ground Truth', color='b')

#plot the interpolated trajectory
plt.plot(interpolated_trajectory_x1(points_to_graph), interpolated_trajectory_x2(points_to_graph), label='Interpolation' ,linestyle='-.', color='r')

#plot the points
plt.scatter(mediciones_df["x1"], mediciones_df["x2"], label="Mediciones", color='g')

plt.title('Trajectory Plot')
plt.xlabel('X Position')
plt.ylabel('Y Position')

plt.legend()
plt.show()