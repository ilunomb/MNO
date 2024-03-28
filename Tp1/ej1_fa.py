import numpy as np
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Definición de las funciones
def fa(x):
    return 0.3 ** (np.abs(x)) * np.sin(4*x) - np.tanh(2*x) + 2


# Puntos de colocación equiespaciados // orden del polinomio
quantity_of_interpolation_points = 10
x_values_for_interpolation = np.linspace(-4, 4, quantity_of_interpolation_points)


# Puntos a graficar de las funciones
points_to_graph = np.linspace(-4, 4, 1000)


# Puntos de colocación Chebyshev // orden del polinomio
quantity_of_interpolation_points_chebyshev = 20


# Generar nodos de Chebyshev en el intervalo [-4, 4]
cheb_nodes = np.polynomial.chebyshev.chebpts1(quantity_of_interpolation_points_chebyshev)
x_values_for_interpolation_chebyshev = 4 * cheb_nodes  # Escalamos los nodos de Chebyshev al intervalo [-4, 4]


# Evaluación de las funciones en los puntos de colocación
fa_values = fa(x_values_for_interpolation)
fCheb_values = fa(x_values_for_interpolation_chebyshev)


# Interpolación utilizando Lagrange para fa(x)
fa_interp_lagrange = lagrange(x_values_for_interpolation, fa_values)
fCheb_interp_lagrange = lagrange(x_values_for_interpolation_chebyshev, fCheb_values)

fa_interp_cubic = CubicSpline(x_values_for_interpolation, fa_values)
fCheb_interp_cubic = CubicSpline(x_values_for_interpolation_chebyshev, fCheb_values)


# Gráficos de las funciones originales y las interpoladas
plt.figure(figsize=(12, 5))


plt.scatter(x_values_for_interpolation, fa_values, label="Puntos de interpolacion equiespaciados", color='g')
plt.plot(points_to_graph, fa(points_to_graph), label='Datos originales', color='r')
plt.plot(points_to_graph, fa_interp_lagrange(points_to_graph), label='Interpolación Lagrange', color='violet',  linestyle='-.')


# Plot cubic spline equiespaciados
plt.plot(points_to_graph, fa_interp_cubic(points_to_graph), label='Interpolación Cubic Spline',  linestyle='--')

plt.title('Interpolación de $f_a(x)$ con Lagrange y Cubic Spline (Equiespaciados)')
plt.xlabel('x')
plt.ylabel('$f_a(x)$')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

plt.scatter(x_values_for_interpolation_chebyshev, fCheb_values, label="Puntos de interpolacion no equiespaciados", color='g')
plt.plot(points_to_graph, fa(points_to_graph), label='Datos originales', color='r')


# Plot cubic spline no equiespaciados
plt.plot(points_to_graph, fCheb_interp_cubic(points_to_graph), label='Interpolación Cubic Spline',  linestyle='--')

plt.title('Interpolación de $f_a(x)$ con Cubic Spline (Chebyshev)')
plt.xlabel('x')
plt.ylabel('$f_a(x)$')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# Gráfico de error máximo en base a cantidad de nodos con Chebyshev y CubicSpline
max_error = []
for i in range(2, 100):
    cheb_nodes = np.polynomial.chebyshev.chebpts1(i)
    x_values_for_interpolation_chebyshev = 4 * cheb_nodes  # Escalamos los nodos de Chebyshev al intervalo [-4, 4]
    fCheb_values = fa(x_values_for_interpolation_chebyshev)
    fCheb_interp_lagrange = CubicSpline(x_values_for_interpolation_chebyshev, fCheb_values)
    max_error.append(np.max(np.abs(fa(points_to_graph) - fCheb_interp_lagrange(points_to_graph))))

plt.plot(range(2, 100), max_error, label='Error maximo', color='r')
plt.title('Error maximo de interpolación de $f_a(x)$ con Cubic Spline en base a la cantidad de nodos (Chebyshev)')
plt.xlabel('Cantidad de nodos')
plt.ylabel('Error maximo')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()