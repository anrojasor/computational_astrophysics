
########################################################################################### ANIMADO

import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D   # para gráficos 3D
from matplotlib import animation

# =============================================================================
# SISTEMA DE UNIDADES:
# - Distancia en UA
# - Tiempo en años
# - Masa en masas solares
#
# Definiciones:
#   G = 4π²  [UA³/(yr²·M☉)]
#   M = 1   [M_solar]
# =============================================================================
G = 4. * np.pi**2      # [UA^3/(yr^2)]
M = 1.0                # [M_solar]

# =============================================================================
# FUNCIONES DEL SISTEMA DE ECUACIONES (3D)
#
# El estado se define como q = [x, y, z, vx, vy, vz]
#
# La ecuación del movimiento es: 
#   d²r/dt² = - (GM / r³)*r     con  r = sqrt(x² + y² + z²)
# =============================================================================
def f_3D(t0, q0):
    """
    f_3D(t0, q0)
    
    Retorna las derivadas del sistema en 3D:
        [dx/dt, dy/dt, dz/dt, ax, ay, az]
    """
    deriv = np.zeros(6)
    # Derivadas de posición:
    deriv[0] = q0[3]   # dx/dt = vx
    deriv[1] = q0[4]   # dy/dt = vy
    deriv[2] = q0[5]   # dz/dt = vz
    
    # Magnitud de r y aceleraciones:
    r2 = q0[0]**2 + q0[1]**2 + q0[2]**2
    r  = np.sqrt(r2)
    deriv[3] = -G * M * q0[0] / (r**3)
    deriv[4] = -G * M * q0[1] / (r**3)
    deriv[5] = -G * M * q0[2] / (r**3)
    return deriv

# =============================================================================
# INTEGRADOR RK4 EN 3D con paso fijo
#
# Cada fila del arreglo solución contendrá:
# [t, x, y, z, vx, vy, vz]
# =============================================================================
def RK4_3D(ODE, t0, q0, tf, n):
    """
    RK4_3D(ODE, t0, q0, tf, n)
    
    Integra el sistema de ODEs en 3D usando el método Runge-Kutta de 4º orden.
    """
    dt = (tf - t0) / (n - 1)
    q = np.zeros((n, len(q0) + 1))
    q[0, 0] = t0
    q[0, 1:] = q0
    for i in range(1, n):
        t_prev = q[i-1, 0]
        q_prev = q[i-1, 1:]
        k1 = dt * ODE(t_prev, q_prev)
        k2 = dt * ODE(t_prev + dt/2, q_prev + k1/2)
        k3 = dt * ODE(t_prev + dt/2, q_prev + k2/2)
        k4 = dt * ODE(t_prev + dt, q_prev + k3)
        q[i, 0] = t_prev + dt
        q[i, 1:] = q_prev + (k1 + 2*k2 + 2*k3 + k4) / 6
    return q

# =============================================================================
# CÁLCULO DE CANTIDADES CONSERVADAS EN 3D:
#
# Se evalúa la energía específica:
#    E = ½ * v² - (GM / r)
# y el módulo del momento angular:
#    L = | r × v |
# =============================================================================
def conserv_quant_3D(q):
    """
    conserv_quant_3D(q)
    
    Calcula la energía específica y el módulo del momento angular del sistema.
    
    Retorna un arreglo CQ con:
        CQ[:,0] = t (años)
        CQ[:,1] = Energía [UA²/yr²]
        CQ[:,2] = |L| [UA²/yr]
    """
    N = len(q)
    CQ = np.zeros((N, 3))
    CQ[:, 0] = q[:, 0]  # tiempo
    v2 = q[:, 4]**2 + q[:, 5]**2 + q[:, 6]**2
    r  = np.sqrt(q[:, 1]**2 + q[:, 2]**2 + q[:, 3]**2)
    E = v2 / 2 - G * M / r
    # Momento angular vectorial, L = r × v:
    Lx = q[:, 2]*q[:, 6] - q[:, 3]*q[:, 5]
    Ly = q[:, 3]*q[:, 4] - q[:, 1]*q[:, 6]
    Lz = q[:, 1]*q[:, 5] - q[:, 2]*q[:, 4]
    L_mod = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    CQ[:, 1] = E
    CQ[:, 2] = L_mod
    return CQ

# =============================================================================
# CONDICIONES INICIALES Y PARÁMETROS DE SIMULACIÓN
#
# Se convierten las condiciones originales a UA y UA/yr:
#   x0 = 4e9 km  --> 4e9 / 1.495978707e8 ≃ 26.73 UA
#   y0 = 0, z0 = 0.
#   vx0 = 0 UA/yr; vy0 ≃ 500*3.15576e7/1.495978707e11 ≃ 0.1055 UA/yr; vz0 = 0.
#
# Puedes cambiar z0 o vz0 si deseas evaluar condiciones 3D generales.
# =============================================================================
x0 = 4e9 / 1.495978707e8               # ≃ 26.73 UA
y0 = 0.0
z0 = 0.0     # Cambiar para condiciones no confinadas en el plano xy.
vx0 = 0.0
vy0 = 500 * 3.15576e7 / 1.495978707e11   # ≃ 0.1055 UA/yr
vz0 = 0.0
Q0 = np.array([x0, y0, z0, vx0, vy0, vz0])

# Parámteros de simulación:
t0_sim = 0.0      # tiempo inicial [años]
tf_sim = 250.0    # tiempo final [años] (aprox. 5 órbitas)
n_steps = 500000  # número de pasos (alta resolución)
dt = (tf_sim - t0_sim) / (n_steps - 1)
print(f"Paso de integración dt = {dt:.2e} años")

# =============================================================================
# EJECUCIÓN DE LA INTEGRACIÓN
# =============================================================================
start = time.time()
Q = RK4_3D(f_3D, t0_sim, Q0, tf_sim, n_steps)
CQ = conserv_quant_3D(Q)
end = time.time()
print('Tiempo de cómputo:', end - start, "segundos")

# =============================================================================
# ANIMACIÓN DE LA TRAYECTORIA EN 3D
# =============================================================================
# Para la animación decimamos los datos usando el factor "step"
step = 1000  # factor de decimación para la animación

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_zlim(-30, 30)
ax.set_xlabel('x [UA]')
ax.set_ylabel('y [UA]')
ax.set_zlabel('z [UA]')
title = ax.set_title('Tiempo = 0.00 años')

# Elementos que se actualizarán en la animación:
line, = ax.plot([], [], [], lw=2, color='cornflowerblue')
# Para el marcador, se pasan listas (uno o más elementos)
point, = ax.plot([], [], [], 'bo', ms=5)
# Se marca la posición del Sol en (0,0,0)
sun, = ax.plot([0], [0], [0], 'ro', ms=8, label='Sol')
ax.legend()

def animate(i):
    idx = i * step
    if idx >= len(Q):
        idx = len(Q) - 1
    # Tomamos los datos hasta el índice idx:
    x_data = Q[:idx, 1]
    y_data = Q[:idx, 2]
    z_data = Q[:idx, 3]
    line.set_data(x_data, y_data)
    line.set_3d_properties(z_data)
    # Para actualizar la posición actual del cometa se pasan listas:
    point.set_data([Q[idx, 1]], [Q[idx, 2]])
    point.set_3d_properties([Q[idx, 3]])
    title.set_text('Tiempo = {:.2f} años'.format(Q[idx, 0]))
    return line, point, title

n_frames = len(Q) // step

anim = animation.FuncAnimation(fig, animate,
                               frames=n_frames, interval=20, blit=True)

# =============================================================================
# Gráficos Adicionales (opcional)
# =============================================================================
# Gráfico: Evolución de la componente x vs. tiempo
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(Q[:, 0], Q[:, 1], color='mediumseagreen', label='x(t)')
ax2.set_title('Evolución de la componente x vs. tiempo')
ax2.set_xlabel('Tiempo [años]')
ax2.set_ylabel('x [UA]')
ax2.legend()
ax2.grid(True)

# Gráfico: Energía y Momento Angular vs. tiempo
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(CQ[:, 0], CQ[:, 1], color='mediumslateblue', label='Energía')
ax3.plot(CQ[:, 0], CQ[:, 2], color='steelblue', label='Momento Angular')
ax3.set_title('Energía y Momento Angular vs. tiempo')
ax3.set_xlabel('Tiempo [años]')
ax3.set_ylabel('Energía [UA²/yr²] y Momento Angular [UA²/yr]')
ax3.legend()
ax3.grid(True)

plt.tight_layout()

# Se muestran las variaciones en las cantidades conservadas:
energia_cambio = np.abs(CQ[-1, 1] - CQ[0, 1])
momento_cambio = np.abs(CQ[-1, 2] - CQ[0, 2])
print('La variación de la energía es:', energia_cambio)
print('La variación del momento angular es:', momento_cambio)

# =============================================================================
# MOSTRAR LA ANIMACIÓN
#
# Al ejecutar este script (.py), se abrirá una ventana interactiva donde se
# reproducirá la animación.
# =============================================================================
plt.show()
