import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import griddata

# Definición de las funciones
def fb(x1, x2):
    return (0.75 * np.exp(-((10*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4) +
            0.65 * np.exp(-((9*x1 + 1)**2)/9 - ((10*x2 + 1)**2)/2) +
            0.55 * np.exp(-((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4) -
            0.01 * np.exp(-((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4))


# Puntos de colocación equiespaciados // orden del polinomio
quantity_of_interpolation_points = 15

# Puntos de colocación Chebyshev // orden del polinomio
quantity_of_interpolation_points_chebyshev = 15

# Generar nodos de Chebyshev en el intervalo [-1, 1]
cheb_nodes_x = np.polynomial.chebyshev.chebpts1(quantity_of_interpolation_points_chebyshev)
cheb_nodes_y = np.polynomial.chebyshev.chebpts1(quantity_of_interpolation_points_chebyshev)

x_values_for_interpolation_chebyshev, y_values_for_interpolation_chebyshev_b2 = np.meshgrid(cheb_nodes_x, cheb_nodes_y)  # Escalamos los nodos de Chebyshev al intervalo [-1, 1]

# Generar puntos equiespaciados en el intervalo [-1, 1]
x_values_for_interpolation, y_values_for_interpolation = np.meshgrid(np.linspace(-1, 1, quantity_of_interpolation_points), np.linspace(-1, 1, quantity_of_interpolation_points_chebyshev))

# Puntos a graficar de las funciones
x_linespace = np.linspace(-1, 1, 1000)
y_linespace = np.linspace(-1, 1, 1000)
x_points_to_graph, y_points_to_graph = np.meshgrid(x_linespace, y_linespace)

# Evaluación de las funciones en los puntos de colocación
fb_values = fb(x_values_for_interpolation, y_values_for_interpolation)
fb_Cheb_values = fb(x_values_for_interpolation_chebyshev, y_values_for_interpolation_chebyshev_b2)

# Interpolación utilizando interp2d para fb(x)
fb_interp = griddata((x_values_for_interpolation.flatten(), y_values_for_interpolation.flatten()), fb_values.flatten(), (x_points_to_graph, y_points_to_graph), method='cubic')
fb_Cheb_interp = griddata((x_values_for_interpolation_chebyshev.flatten(), y_values_for_interpolation_chebyshev_b2.flatten()), fb_Cheb_values.flatten(), (x_points_to_graph, y_points_to_graph), method='cubic')

# Gráficos de las funciones originales y las interpoladas
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(projection='3d')
ax.scatter(x_values_for_interpolation_chebyshev, y_values_for_interpolation_chebyshev_b2, fb_Cheb_values, label="Puntos de interpolación Chebyshev", color='g')
ax.plot_surface(x_points_to_graph, y_points_to_graph, fb(x_points_to_graph, y_points_to_graph), cmap=plt.cm.viridis, alpha=0.9)
ax.plot_wireframe(x_points_to_graph, y_points_to_graph, fb_Cheb_interp, color="black", alpha=0.3)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_b(x, y)$')
ax.set_title('Interpolación de $f_b(x, y)$ con Bicubic Interpolation (Chebyshev)')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(projection='3d')
ax.scatter(x_values_for_interpolation.flatten(), y_values_for_interpolation.flatten(), fb_values.flatten(), label="Puntos de interpolación equiespaciados", color='g')
ax.plot_surface(x_points_to_graph, y_points_to_graph, fb(x_points_to_graph, y_points_to_graph), cmap=plt.cm.viridis, alpha=0.9)
ax.plot_wireframe(x_points_to_graph, y_points_to_graph, fb_interp, color="black", alpha=0.3)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_b(x, y)$')
ax.set_title('Interpolación de $f_b(x, y)$ con Bicubic Interpolation (Equiespaciado)')

plt.tight_layout()
plt.show()

# Gráfico de error máximo en base a la cantidad de puntos de interpolación (Equiespaciados)
max_error = []
for i in range(4, 50):
    x_values_for_interpolation, y_values_for_interpolation = np.meshgrid(np.linspace(-1, 1, i), np.linspace(-1, 1, i))
    fb_values = fb(x_values_for_interpolation, y_values_for_interpolation)
    fb_interp = griddata((x_values_for_interpolation.flatten(), y_values_for_interpolation.flatten()), fb_values.flatten(), (x_points_to_graph, y_points_to_graph), method='cubic')
    max_error.append(np.max(np.abs(fb(x_points_to_graph, y_points_to_graph) - fb_interp)))

plt.figure(figsize=(12, 6))
plt.plot(range(4, 50), max_error, label='Error maximo', color='r')
plt.title('Error máximo en función de la cantidad de puntos de interpolación (Equiespaciados)')
plt.xlabel('Cantidad de puntos de interpolación')
plt.ylabel('Error máximo')
plt.grid()
plt.legend()



plt.tight_layout()
plt.show()

# Gráfico de error máximo en base a la cantidad de puntos de interpolación (Chebyshev)
max_error = []

for i in range(4, 50):
    x_values_for_interpolation_chebyshev, y_values_for_interpolation_chebyshev_b2 = np.meshgrid(np.linspace(-1, 1, i), np.linspace(-1, 1, i))
    fb_values = fb(x_values_for_interpolation_chebyshev, y_values_for_interpolation_chebyshev_b2)
    fb_interp = griddata((x_values_for_interpolation_chebyshev.flatten(), y_values_for_interpolation_chebyshev_b2.flatten()), fb_values.flatten(), (x_points_to_graph, y_points_to_graph), method='cubic')
    max_error.append(np.max(np.abs(fb(x_points_to_graph, y_points_to_graph) - fb_interp)))

plt.figure(figsize=(12, 6))
plt.plot(range(4, 50), max_error, label='Error maximo', color='r')
plt.title('Error máximo en función de la cantidad de puntos de interpolación (Chebyshev)')
plt.xlabel('Cantidad de puntos de interpolación')
plt.ylabel('Error máximo')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()