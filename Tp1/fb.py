import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# Definición de las funciones
def fb(x1, x2):
    return (0.75 * np.exp(-((10*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4) +
            0.65 * np.exp(-((9*x1 + 1)**2)/9 - ((10*x2 + 1)**2)/2) +
            0.55 * np.exp(-((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4) -
            0.01 * np.exp(-((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4))

# Puntos de colocación Chebyshev // orden del polinomio
n_points_b = 15

# Generar nodos de Chebyshev en el intervalo [-1, 1]
cheb_nodes_b1 = np.polynomial.chebyshev.chebpts1(n_points_b)
cheb_nodes_b2 = np.polynomial.chebyshev.chebpts1(n_points_b)

x_Cheb_b1, x_Cheb_b2 = np.meshgrid(cheb_nodes_b1, cheb_nodes_b2)  # Escalamos los nodos de Chebyshev al intervalo [-1, 1]

# Generar puntos equiespaciados en el intervalo [-1, 1]
x_b1, x_b2 = np.meshgrid(np.linspace(-1, 1, n_points_b), np.linspace(-1, 1, n_points_b))

# puntos a graficar de las funciones
x1_Vb = np.linspace(-1, 1, 100)
x2_Vb = np.linspace(-1, 1, 100)
x_b1_V, x_b2_V = np.meshgrid(x1_Vb, x2_Vb)

# Evaluación de las funciones en los puntos de colocación
fb_values = fb(x_b1, x_b2)
fb_Cheb_values = fb(x_Cheb_b1, x_Cheb_b2)

# Interpolación utilizando interp2d para fb(x)
fb_interp = interp2d(x_b1.flatten(), x_b2.flatten(), fb_values, kind='linear')
fb_Cheb_interp = interp2d(cheb_nodes_b1, cheb_nodes_b2, fb_Cheb_values, kind='linear')

# Gráficos de las funciones originales y las interpoladas
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(x_Cheb_b1, x_Cheb_b2, fb_Cheb_values, label="Puntos de interpolación Chebyshev", color='g')
ax.plot_surface(x_b1_V, x_b2_V, fb(x_b1_V, x_b2_V), cmap=plt.cm.cividis, alpha=0.5)
ax.plot_wireframe(x_b1_V, x_b2_V, fb_Cheb_interp(x1_Vb, x2_Vb), cmap=plt.cm.cividis, alpha=0.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f_b(x)$')
ax.set_title('Interpolación de $f_b(x)$ con interp2d (Chebyshev)')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(x_b1.flatten(), x_b2.flatten(), fb_values.flatten(), label="Puntos de interpolación equiespaciados", color='g')
ax.plot_surface(x_b1_V, x_b2_V, fb(x_b1_V, x_b2_V), cmap=plt.cm.cividis, alpha=0.5)
ax.plot_wireframe(x_b1_V, x_b2_V, fb_interp(x1_Vb, x2_Vb), cmap=plt.cm.cividis, alpha=0.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f_b(x)$')
ax.set_title('Interpolación de $f_b(x)$ con interp2d (Equiespaciado)')

plt.show()
