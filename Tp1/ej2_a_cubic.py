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
points_for_error = np.linspace(mediciones_df.index.min(), mediciones_df.index.max(), len(ground_truth_df))

#error min and max between the interpolated trajectory and the ground truth 
error_mean = np.mean(np.sqrt((interpolated_trajectory_x1(points_for_error) - ground_truth_df["x1"])**2 + (interpolated_trajectory_x2(points_for_error) - ground_truth_df["x2"])**2))
error_max = np.max(np.sqrt((interpolated_trajectory_x1(points_for_error) - ground_truth_df["x1"])**2 + (interpolated_trajectory_x2(points_for_error) - ground_truth_df["x2"])**2))

print(f"Error min: {error_mean}")
print(f"Error max: {error_max}")






#plot init
plt.figure(figsize=(12, 5))

#plot the ground truth
plt.plot(ground_truth_df["x1"], ground_truth_df["x2"], label='Trayectoria real', color='b')

#plot the interpolated trajectory
plt.plot(interpolated_trajectory_x1(points_to_graph), interpolated_trajectory_x2(points_to_graph), label='Interpolación' ,linestyle='-.', color='r')

#plot the points
plt.scatter(mediciones_df["x1"], mediciones_df["x2"], label="Mediciones", color='g')

plt.title('Trajectoria del tractor con Cubic Spline')
plt.xlabel('X')
plt.ylabel('Y')

plt.grid()

plt.legend()
plt.show()