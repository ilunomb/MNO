import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Cargar datos de los archivos CSV
mediciones_df = pd.read_csv("mnyo_mediciones.csv", sep=" ", header=None, names=["x1", "x2"])
mediciones_2_df = pd.read_csv("mnyo_mediciones2.csv", sep=" ", header=None, names=["x1", "x2"])
ground_truth_df = pd.read_csv("mnyo_ground_truth.csv", sep=" ", header=None, names=["x1", "x2"])

#generate 1000 points to graph
points_to_graph = np.linspace(mediciones_df.index.min(), mediciones_df.index.max(), 1000)
points_to_graph_v2 = np.linspace(mediciones_2_df.index.min(), mediciones_2_df.index.max(), 1000)

#interpolate the trajectory based on index
interpolated_trajectory_v1_x1 = CubicSpline(mediciones_df.index, mediciones_df["x1"])
interpolated_trajectory_v1_x2 = CubicSpline(mediciones_df.index, mediciones_df["x2"])

interpolated_trajectory_v2_x1 = CubicSpline(mediciones_2_df.index, mediciones_2_df["x1"])
interpolated_trajectory_v2_x2 = CubicSpline(mediciones_2_df.index, mediciones_2_df["x2"])


# Extraer coordenadas x1 y x2 de cada vehículo
x1_vehiculo1, x2_vehiculo1 = mediciones_df["x1"], mediciones_df["x2"]
x1_vehiculo2, x2_vehiculo2 = mediciones_2_df["x1"], mediciones_2_df["x2"]

# Interpolación de la trayectoria del segundo vehículo con Lagrange
polinomio_interpolado = CubicSpline(x1_vehiculo2, x2_vehiculo2)
x1_interpolados = np.linspace(min(x1_vehiculo2), max(x1_vehiculo2), len(x1_vehiculo1))
x2_interpolados = polinomio_interpolado(x1_interpolados)



def newton_raphson_doble_variable(f1, f2, x0, y0, tolerancia=1e-6, max_iter=1000):
         # Calculo el jacobiano 
         def jacobiano(x, y): 
            #  j11 = (f1(x + tolerancia, y) - f1(x, y)) / tolerancia 
            #  j12 = (f1(x, y + tolerancia) - f1(x, y)) / tolerancia 
            #  j21 = (f2(x + tolerancia, y) - f2(x, y)) / tolerancia 
            #  j22 = (f2(x, y + tolerancia) - f2(x, y)) / tolerancia 
             return np.array(
                    [[(f1(x + tolerancia, y) - f1(x, y)) / tolerancia, (f1(x, y + tolerancia) - f1(x, y)) / tolerancia ], 
                     [(f2(x + tolerancia, y) - f2(x, y)) / tolerancia, (f2(x, y + tolerancia) - f2(x, y)) / tolerancia ]
            ]) 
  
         for i in range(max_iter): 
             j_inv = np.linalg.inv(jacobiano(x0, y0)) 
             f = np.array([f1(x0, y0), f2(x0, y0)]) 
             p = np.array([x0, y0]) - j_inv @ f 
             if np.linalg.norm(p - np.array([x0, y0])) < tolerancia: 
                 return x0 
             x0, y0 = p 
         return None
  
def f1(x, y): 
         return interpolated_trajectory_v1_x1(x) - interpolated_trajectory_v2_x1(y) 
  
def f2(x, y): 
         return interpolated_trajectory_v1_x2(x) - interpolated_trajectory_v2_x2(y) 
  
t_intersect = newton_raphson_doble_variable(f1, f2, 0, 0)

# Calculate the intersection point 
m1_x1_intersect = interpolated_trajectory_v1_x1(t_intersect) 
m1_x2_intersect = interpolated_trajectory_v1_x2(t_intersect)

print(f"El punto de la intersección es en: ({m1_x1_intersect}, {m1_x2_intersect})")




# Graficar trayectorias de ambos vehículos
plt.figure(figsize=(10, 6))
plt.plot(ground_truth_df["x1"], ground_truth_df["x2"], label='Ground Truth', color='b')
plt.plot(interpolated_trajectory_v1_x1(points_to_graph), interpolated_trajectory_v1_x2(points_to_graph), label='Interpolación v1' ,linestyle='-.', color='r')
plt.plot(interpolated_trajectory_v2_x1(points_to_graph_v2), interpolated_trajectory_v2_x2(points_to_graph_v2), label='Interpolación v2' ,linestyle='-.', color='g')
plt.scatter(m1_x1_intersect, m1_x2_intersect, color='r', label="Intersección")

# Configuración de la gráfica
plt.xlabel("Coordenada x1")
plt.ylabel("Coordenada x2")
plt.title("Trayectorias de Vehículos con Cubic Spline")
plt.legend()
plt.grid()
plt.show()
