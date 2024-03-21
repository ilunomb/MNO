import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns

# Definición de las funciones
def fa(x):
    return 0.3 ** (np.abs(x)) * np.sin(4*x) - np.tanh(2*x) + 2

# Puntos de colocación equiespaciados // orden del polinomio
quantity_of_interpolation_points = 10

# Puntos de colocación Chebyshev // orden del polinomio
quantity_of_interpolation_points_chebyshev = 10

x_values_for_interpolation = np.linspace(-4, 4, quantity_of_interpolation_points)

# puntos a graficar de las funciones
points_to_graph = np.linspace(-4, 4, 1000)

# Generar nodos de Chebyshev en el intervalo [-4, 4]
cheb_nodes = np.polynomial.chebyshev.chebpts1(quantity_of_interpolation_points_chebyshev)
x_values_for_interpolation_chebyshev = 4 * cheb_nodes  # Escalamos los nodos de Chebyshev al intervalo [-4, 4]

# Evaluación de las funciones en los puntos de colocación
fa_values = fa(x_values_for_interpolation)
fCheb_values = fa(x_values_for_interpolation_chebyshev)

# Interpolación utilizando Lagrange para fa(x)
fa_interp = CubicSpline(x_values_for_interpolation, fa_values)
fCheb_interp = CubicSpline(x_values_for_interpolation_chebyshev, fCheb_values)

# Gráficos de las funciones originales y las interpoladas
plt.figure(figsize=(12, 5))

plt.subplot(2, 2, 1)
plt.scatter(x_values_for_interpolation, fa_values, label="Puntos de interpolacion equiespaciados", color='g')
plt.plot(points_to_graph, fa(points_to_graph), label='Datos originales', color='r')
plt.plot(points_to_graph, fa_interp(points_to_graph), label='Interpolación',  linestyle='-.')
plt.title('Interpolación de $f_a(x)$ con Cubic Spline')
plt.xlabel('x')
plt.ylabel('$f_a(x)$')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.scatter(x_values_for_interpolation_chebyshev, fCheb_values, label="Puntos de interpolacion no equiespaciados", color='g')
plt.plot(points_to_graph, fa(points_to_graph), label='Datos originales', color='r')
plt.plot(points_to_graph, fCheb_interp(points_to_graph), label='Interpolación',  linestyle='-.')
plt.title('Interpolación de $f_a(x)$ con Cubic Spline (Chebyshev)')
plt.xlabel('x')
plt.ylabel('$f_a(x)$')
plt.legend()
plt.grid()

# Gráficos de los errores de interpolación
plt.subplot(2, 2, 3)
plt.plot(points_to_graph, np.abs(fa(points_to_graph) - fa_interp(points_to_graph)), label='Error equiespaciados', color='r')
plt.title('Error de interpolación de $f_a(x)$ con Cubic Spline')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.grid()


plt.subplot(2, 2, 4)
plt.plot(points_to_graph, np.abs(fa(points_to_graph) - fCheb_interp(points_to_graph)), label='Error no equiespaciados', color='r')
plt.title('Error de interpolación de $f_a(x)$ con Cubic Spline (Chebyshev)')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
