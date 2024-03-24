import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import newton



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



# WARNING SECCIÓN INTERSECCIÓN 

# def diff(x):
#     return np.interp(x, x1_vehiculo1, x2_vehiculo1) - np.interp(x, x1_interpolados, x2_interpolados)

# # Usar el método de Newton para encontrar la intersección
# x_interseccion = newton(diff, x0=0)

# # Calcular las coordenadas de la intersección
# y_interseccion = np.interp(x_interseccion, x1_vehiculo1, x2_vehiculo1)

# # Mostrar las coordenadas de la intersección
# print(f"Las coordenadas de la intersección son: ({x_interseccion}, {y_interseccion})")


def newton_raphson_doble_variable(f1, f2, x0, y0, tol=1e-6, max_iter=1000): 
         # p(n-1)=p(n)-(jacobiano*-1)(p(n)) * f(p(n)) 
         # f1, f2: funciones f1(x, y), f2(x, y) 
  
         # Calculo el jacobiano 
         def jacobiano(x, y): 
             j11 = (f1(x + tol, y) - f1(x, y)) / tol 
             j12 = (f1(x, y + tol) - f1(x, y)) / tol 
             j21 = (f2(x + tol, y) - f2(x, y)) / tol 
             j22 = (f2(x, y + tol) - f2(x, y)) / tol 
             return np.array([[j11, j12], [j21, j22]]) 
  
         for _ in range(max_iter): 
             j_inv = np.linalg.inv(jacobiano(x0, y0)) 
             f = np.array([f1(x0, y0), f2(x0, y0)]) 
             p = np.array([x0, y0]) - j_inv @ f 
             if np.linalg.norm(p - np.array([x0, y0])) < tol: 
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

print(f"The intersection point is at ({m1_x1_intersect}, {m1_x2_intersect})")


# TERMINA SECCIÓN INTERSECCIÓN


# Graficar trayectorias de ambos vehículos
plt.figure(figsize=(10, 6))
plt.plot(ground_truth_df["x1"], ground_truth_df["x2"], label='Ground Truth', color='b')
plt.plot(interpolated_trajectory_v1_x1(points_to_graph), interpolated_trajectory_v1_x2(points_to_graph), label='Interpolation v1' ,linestyle='-.', color='r')
plt.plot(interpolated_trajectory_v2_x1(points_to_graph_v2), interpolated_trajectory_v2_x2(points_to_graph_v2), label='Interpolation v2' ,linestyle='-.', color='g')
# plt.plot(x1_vehiculo1, x2_vehiculo1, label="Trayectoria Vehículo 1", marker='o')
# plt.plot(x1_vehiculo2, x2_vehiculo2, label="Trayectoria Vehículo 2", marker='o')
# plt.plot(x1_interpolados, x2_interpolados, label="Trayectoria Interpolada Vehículo 2")
# plt.plot(x_interseccion, y_interseccion, 'ro', label="Intersección")
plt.plot(m1_x1_intersect, m1_x2_intersect, 'ro', label="Intersección")

# Configuración de la gráfica
plt.xlabel("Coordenada x1")
plt.ylabel("Coordenada x2")
plt.title("Trayectorias de Vehículos")
plt.legend()
plt.grid()
plt.show()



"""
Explicación linea 31 del diff(x)

Sí, estás en lo correcto. La función np.interp(x, x1_interpolados, x2_interpolados) está realizando 
una interpolación sobre los datos ya interpolados.

Esto puede ser útil en ciertos casos, por ejemplo, si necesitas interpolar a un conjunto de puntos x 
que no estaban presentes en el conjunto original de puntos x utilizados para la primera interpolación. 
Sin embargo, si los puntos x a los que estás interpolando ya estaban presentes en el conjunto original 
de puntos x, entonces esta operación puede no ser necesaria y podría introducir errores adicionales 
debido a la naturaleza aproximada de la interpolación.

Es importante tener en cuenta que la interpolación es una técnica de estimación que puede introducir 
errores, especialmente cuando se interpola fuera del rango de los datos originales. Por lo tanto, 
siempre es una buena idea entender completamente tus datos y tus necesidades antes de decidir si la 
interpolación es la técnica correcta a utilizar




Sí, en tu código estás calculando la diferencia entre dos interpolaciones en el punto x. 
La primera interpolación se realiza utilizando los datos originales del vehículo 1
, mientras que la segunda interpolación se realiza utilizando los datos interpolados del vehículo 2

Calcula la diferencia entre las posiciones estimadas del vehículo 1 y las posiciones interpoladas 
del vehículo 2 en el punto x. Esto puede tener sentido en algunos contextos, como por ejemplo, si 
estás buscando puntos de intersección entre las trayectorias de ambos vehículos.

Sin embargo, debes tener en cuenta que este enfoque asume que las trayectorias de ambos vehículos 
son comparables en términos de escalas, direcciones y velocidades. Además, la calidad de la 
interpolación y la precisión de los resultados dependerán de la calidad y cantidad de datos 
disponibles para cada vehículo.




"""