import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import seaborn as sns

# Definición de las funciones
def fa(x):
    return 0.3 ** (np.abs(x)) * np.sin(4*x) - np.tanh(2*x) + 2

# Puntos de colocación equiespaciados // orden del polinomio
quantity_of_interpolation_points1 = 10
quantity_of_interpolation_points2 = 15
quantity_of_interpolation_points3 = 20

# Puntos de colocación Chebyshev // orden del polinomio
quantity_of_interpolation_points_chebyshev1 = 10
quantity_of_interpolation_points_chebyshev2 = 15
quantity_of_interpolation_points_chebyshev3 = 20

x_values_for_interpolation1 = np.linspace(-4, 4, quantity_of_interpolation_points1)
x_values_for_interpolation2 = np.linspace(-4, 4, quantity_of_interpolation_points2)
x_values_for_interpolation3 = np.linspace(-4, 4, quantity_of_interpolation_points3)

# puntos a graficar de las funciones
points_to_graph = np.linspace(-4, 4, 1000)

# Generar nodos de Chebyshev en el intervalo [-4, 4]
cheb_nodes1 = np.polynomial.chebyshev.chebpts1(quantity_of_interpolation_points_chebyshev1)
cheb_nodes2 = np.polynomial.chebyshev.chebpts1(quantity_of_interpolation_points_chebyshev2)
cheb_nodes3 = np.polynomial.chebyshev.chebpts1(quantity_of_interpolation_points_chebyshev3)

# Escalamos los nodos de Chebyshev al intervalo [-4, 4]
x_values_for_interpolation_chebyshev1 = 4 * cheb_nodes1  
x_values_for_interpolation_chebyshev2 = 4 * cheb_nodes2
x_values_for_interpolation_chebyshev3 = 4 * cheb_nodes3

# Evaluación de las funciones en los puntos de colocación
fa_values1 = fa(x_values_for_interpolation1)
fa_values2 = fa(x_values_for_interpolation2)
fa_values3 = fa(x_values_for_interpolation3)
fCheb_values1 = fa(x_values_for_interpolation_chebyshev1)
fCheb_values2 = fa(x_values_for_interpolation_chebyshev2)
fCheb_values3 = fa(x_values_for_interpolation_chebyshev3)

# Interpolación utilizando Lagrange para fa(x)
fa_interp1 = lagrange(x_values_for_interpolation1, fa_values1)
fa_interp2 = lagrange(x_values_for_interpolation2, fa_values2)
fa_interp3 = lagrange(x_values_for_interpolation3, fa_values3)
fCheb_interp1 = lagrange(x_values_for_interpolation_chebyshev1, fCheb_values1)
fCheb_interp2 = lagrange(x_values_for_interpolation_chebyshev2, fCheb_values2)
fCheb_interp3 = lagrange(x_values_for_interpolation_chebyshev3, fCheb_values3)

# Gráficos de las funciones originales y las interpoladas
plt.figure(figsize=(12, 5))

ax = plt.axes()

plt.plot(points_to_graph, fa(points_to_graph), label='Datos originales', color='black')

# plt.subplot(1, 2, 1)
plt.scatter(x_values_for_interpolation1, fa_values1, label="Puntos de interpolacion equiespaciados n = 10", color='r')
plt.scatter(x_values_for_interpolation2, fa_values2, label="Puntos de interpolacion equiespaciados n = 15", color='g')
plt.scatter(x_values_for_interpolation3, fa_values3, label="Puntos de interpolacion equiespaciados n = 20", color='b')
plt.plot(points_to_graph, fa_interp1(points_to_graph), label='Interpolación n = 10', color='r', linestyle='-.')
plt.plot(points_to_graph, fa_interp2(points_to_graph), label='Interpolación n = 15', color='g', linestyle='-.')
plt.plot(points_to_graph, fa_interp3(points_to_graph), label='Interpolación n = 20', color='b', linestyle='-.')
plt.title('Interpolación de $f_a(x)$ con Lagrange')
plt.xlabel('x')
plt.ylabel('$f_a(x)$')
# plt.legend()
# plt.grid()

# plt.subplot(2, 2, 2)
plt.scatter(x_values_for_interpolation_chebyshev1, fCheb_values1, label="Puntos de interpolacion no equiespaciados n = 10", color='yellow')
plt.scatter(x_values_for_interpolation_chebyshev2, fCheb_values2, label="Puntos de interpolacion no equiespaciados n = 15", color='purple')
plt.scatter(x_values_for_interpolation_chebyshev3, fCheb_values3, label="Puntos de interpolacion no equiespaciados n = 20", color='orange')
# plt.plot(points_to_graph, fa(points_to_graph), label='Datos originales', color='r')
plt.plot(points_to_graph, fCheb_interp1(points_to_graph), label='Interpolación n = 10',  linestyle='-.', color='yellow')
plt.plot(points_to_graph, fCheb_interp2(points_to_graph), label='Interpolación n = 15',  linestyle='-.', color='purple')
plt.plot(points_to_graph, fCheb_interp3(points_to_graph), label='Interpolación n = 20',  linestyle='-.', color='orange')
plt.title('Interpolación de $f_a(x)$ con Lagrange (Chebyshev)')
plt.xlabel('x')
plt.ylabel('$f_a(x)$')

plt.ylim(0, 4)

#make background gray
ax.set_facecolor('lightgray')

plt.legend()
plt.grid()

# # Gráficos de los errores de interpolación
# plt.subplot(2, 2, 3)
# plt.plot(points_to_graph, np.abs(fa(points_to_graph) - fa_interp1(points_to_graph)), label='Error equiespaciados', color='r')
# plt.title('Error de interpolación de $f_a(x)$ con Cubic Spline')
# plt.xlabel('x')
# plt.ylabel('Error')
# plt.legend()
# plt.grid()


# plt.subplot(2, 2, 4)
# plt.plot(points_to_graph, np.abs(fa(points_to_graph) - fCheb_interp1(points_to_graph)), label='Error no equiespaciados', color='r')
# plt.title('Error de interpolación de $f_a(x)$ con Cubic Spline (Chebyshev)')
# plt.xlabel('x')
# plt.ylabel('Error')
# plt.legend()
# plt.grid()

plt.tight_layout()
plt.show()
    