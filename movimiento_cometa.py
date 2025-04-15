import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  # para gráficos 3D

# =============================================================================
# Constantes y sistema de unidades:
# - Distancia en UA
# - Tiempo en años
# - Masa en masas solares
#
# Usamos: G = 4π² [UA³/(yr²·M☉)] y M = 1.
# =============================================================================
G = 4. * np.pi**2      # [UA³/(yr²)]
M = 1.0                # [M_solar]

# =============================================================================
# Función que define el sistema de ecuaciones en 3D:
# q = [x, y, z, vx, vy, vz]
# Ecuación: d²r/dt² = - (G M)/(r³) · r, con r = sqrt(x² + y² + z²)
# =============================================================================
def f_3D(t, q):
    deriv = np.zeros(6)
    # Derivadas de posición:
    deriv[0] = q[3]  # dx/dt = vx
    deriv[1] = q[4]  # dy/dt = vy
    deriv[2] = q[5]  # dz/dt = vz
    r2 = q[0]**2 + q[1]**2 + q[2]**2
    r = np.sqrt(r2)
    # Derivadas de velocidad (aceleraciones):
    deriv[3] = -G * M * q[0] / (r**3)
    deriv[4] = -G * M * q[1] / (r**3)
    deriv[5] = -G * M * q[2] / (r**3)
    return deriv

# =============================================================================
# Integración mediante RK4 de 4º orden con paso fijo en 3D.
# Se almacena la solución en un arreglo con [t, x, y, z, vx, vy, vz] en cada fila.
# =============================================================================
def RK4_3D(ODE, t0, q0, tf, n):
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
        q[i, 1:] = q_prev + (k1 + 2*k2 + 2*k3 + k4)/6
    return q

# =============================================================================
# Función para calcular cantidades conservadas en 3D:
# Energía: E = ½*v² - (G M)/r
# Momento Angular: L = |r x v|
# =============================================================================
def conserv_quant_3D(q):
    N = len(q)
    CQ = np.zeros((N, 3))
    CQ[:, 0] = q[:, 0]  # tiempo
    v2 = q[:, 4]**2 + q[:, 5]**2 + q[:, 6]**2
    r = np.sqrt(q[:, 1]**2 + q[:, 2]**2 + q[:, 3]**2)
    E = v2/2 - G * M / r
    # Cálculo del módulo del momento angular: L = |r x v|
    Lx = q[:, 2]*q[:, 6] - q[:, 3]*q[:, 5]
    Ly = q[:, 3]*q[:, 4] - q[:, 1]*q[:, 6]
    Lz = q[:, 1]*q[:, 5] - q[:, 2]*q[:, 4]
    L_mod = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    CQ[:, 1] = E
    CQ[:, 2] = L_mod
    return CQ

# =============================================================================
# Condiciones iniciales (convertidas a UA y UA/yr)
# =============================================================================
x0 = 4e9 / 1.495978707e8         # ≃ 26.73 UA
y0 = 0.0
z0 = 0.0
vx0 = 0.0
vy0 = 500 * 3.15576e7 / 1.495978707e11  # ≃ 0.1055 UA/yr
vz0 = 0.0
Q0 = np.array([x0, y0, z0, vx0, vy0, vz0])

# =============================================================================
# PARÁMETROS PARA INTEGRACIÓN CON PASO FIJO
# =============================================================================
t0_sim = 0.0         # tiempo inicial [años]
tf_sim = 250.0       # tiempo final [años] (aproximadamente 5 órbitas)
n_steps = 500000     # número de pasos
dt_fixed = (tf_sim - t0_sim) / (n_steps - 1)
print(f"Integración con RK4 de paso fijo: dt = {dt_fixed:.2e} años")

# =============================================================================
# EJECUCIÓN DE LA INTEGRACIÓN CON PASO FIJO
# =============================================================================
start = time.time()
Q_fixed = RK4_3D(f_3D, t0_sim, Q0, tf_sim, n_steps)
CQ_fixed = conserv_quant_3D(Q_fixed)
end = time.time()
print("Tiempo de cómputo (RK4 paso fijo):", end - start, "segundos")

# =============================================================================
# GRÁFICOS PARA EL MÉTODO DE PASO FIJO
# =============================================================================

# 1. Gráfico 3D de la órbita (Paso fijo)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(Q_fixed[:, 1], Q_fixed[:, 2], Q_fixed[:, 3],
        color='cornflowerblue', lw=0.8, label=f'dt = {dt_fixed:.2e} yr')
ax.set_title('Trayectoria 3D del cometa (Paso Fijo)')
ax.set_xlabel('x [UA]')
ax.set_ylabel('y [UA]')
ax.set_zlabel('z [UA]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 2. Gráfico de x vs. t (Paso fijo)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(Q_fixed[:, 0], Q_fixed[:, 1], color='mediumseagreen', label='x(t)')
ax.set_title('Evolución de x vs. t (Paso Fijo)')
ax.set_xlabel('Tiempo [años]')
ax.set_ylabel('x [UA]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 3. Gráfico de y vs. t (Paso fijo)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(Q_fixed[:, 0], Q_fixed[:, 2], color='darkorange', label='y(t)')
ax.set_title('Evolución de y vs. t (Paso Fijo)')
ax.set_xlabel('Tiempo [años]')
ax.set_ylabel('y [UA]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 4. Gráfico de Energía y Momento Angular vs. t (Paso fijo)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(CQ_fixed[:, 0], CQ_fixed[:, 1], color='mediumslateblue', label='Energía')
ax.plot(CQ_fixed[:, 0], CQ_fixed[:, 2], color='steelblue', label='Momento Angular')
ax.set_title('Energía y Momento Angular vs. t (Paso Fijo)')
ax.set_xlabel('Tiempo [años]')
ax.set_ylabel('Energía [UA²/yr²] / Momento Angular [UA²/yr]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()


# =============================================================================
# MÉTODO RK4 ADAPTATIVO
# =============================================================================
def ARK4(ODE, t0, q0, tf, n, dt=1e-2, epsilon=1e-8):
    """
    Método de Runge-Kutta de 4º orden con paso adaptativo.
    
    Parámetros:
      ODE     : función que define el sistema (por ejemplo, f_3D)
      t0      : tiempo inicial
      q0      : condiciones iniciales (vector de estado)
      tf      : tiempo final
      n       : número máximo de pasos permitidos
      dt      : tamaño inicial del paso
      epsilon : tolerancia requerida para el error en posición
    Retorna:
      q       : arreglo con la solución [t, x, y, z, vx, vy, vz]
    """
    q = np.zeros((n, len(q0)+1))
    q[0, 0] = t0
    q[0, 1:] = q0
    
    def ks(t, y, dt):
        k1 = dt * ODE(t, y)
        k2 = dt * ODE(t + dt/2, y + k1/2)
        k3 = dt * ODE(t + dt/2, y + k2/2)
        k4 = dt * ODE(t + dt, y + k3)
        return (k1 + 2*k2 + 2*k3 + k4) / 6
    
    i = 0
    current_t = t0
    while i < n - 2 and current_t < tf:
        if current_t + 2*dt > tf:
            dt = (tf - current_t) / 2
        
        # Dos pasos pequeños de tamaño dt
        t1 = current_t + dt
        y_temp = q[i, 1:] + ks(current_t, q[i, 1:], dt)
        t2 = current_t + 2*dt
        y_small = y_temp + ks(t1, y_temp, dt)
        
        # Un único paso grande de tamaño 2*dt
        y_big = q[i, 1:] + ks(current_t, q[i, 1:], 2*dt)
        
        # Estimación del error usando las componentes de posición x e y
        error = np.sqrt((y_small[0] - y_big[0])**2 + (y_small[1] - y_big[1])**2)
        Theta = error / (30 * dt * epsilon)
        
        if Theta < 1:
            q[i+1, 0] = t1
            q[i+1, 1:] = y_temp
            q[i+2, 0] = t2
            q[i+2, 1:] = y_small
            current_t = t2
            i += 2
            dt = dt * (Theta ** (-0.25))
        else:
            dt = dt * (Theta ** (-0.25))
            continue
    Q_adapt = q[:i+1]
    print(f"Integración adaptativa: t = {current_t:.2f} años con {i+1} pasos.")
    return Q_adapt

# Parámetros para el método adaptativo
n_adapt = 100000    # número máximo de pasos
dt0 = 1e-2          # tamaño inicial del paso
epsilon = 1e-8      # tolerancia

print("\nEjecutando RK4 adaptativo:")
start_adapt = time.time()
Q_adapt = ARK4(f_3D, t0_sim, Q0, tf_sim, n_adapt, dt=dt0, epsilon=epsilon)
CQ_adapt = conserv_quant_3D(Q_adapt)
end_adapt = time.time()
print("Tiempo de cómputo (RK4 adaptativo):", end_adapt - start_adapt, "segundos")

# =============================================================================
# GRÁFICOS PARA EL MÉTODO ADAPTATIVO
# =============================================================================

# 1. Gráfico 3D de la órbita del cometa (adaptativo)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(Q_adapt[:, 1], Q_adapt[:, 2], Q_adapt[:, 3],
        'o-', markersize=3, color='darkblue', label='Trayectoria Adaptativa')
ax.set_title('Trayectoria 3D del cometa (RK4 Adaptativo)')
ax.set_xlabel('x [UA]')
ax.set_ylabel('y [UA]')
ax.set_zlabel('z [UA]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 2. Gráfico de x vs. t (adaptativo)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(Q_adapt[:, 0], Q_adapt[:, 1], 'o-', markersize=3, color='orangered', label='x(t)')
ax.set_title('Evolución de x vs. t (RK4 Adaptativo)')
ax.set_xlabel('Tiempo [años]')
ax.set_ylabel('x [UA]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 3. Gráfico de y vs. t (adaptativo)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(Q_adapt[:, 0], Q_adapt[:, 2], 'o-', markersize=3, color='darkgreen', label='y(t)')
ax.set_title('Evolución de y vs. t (RK4 Adaptativo)')
ax.set_xlabel('Tiempo [años]')
ax.set_ylabel('y [UA]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 4. Gráfico de Energía y Momento Angular vs. t (adaptativo)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(CQ_adapt[:, 0], CQ_adapt[:, 1], color='mediumslateblue', label='Energía')
ax.plot(CQ_adapt[:, 0], CQ_adapt[:, 2], color='steelblue', label='Momento Angular')
ax.set_title('Energía y Momento Angular vs. t (RK4 Adaptativo)')
ax.set_xlabel('Tiempo [años]')
ax.set_ylabel('Energía [UA²/yr²] / Momento Angular [UA²/yr]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 5. Gráfico de x vs. t con líneas verticales en cada paso (adaptativo)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(Q_adapt[:, 0], Q_adapt[:, 1], 'o-', markersize=3, color='orangered', label='x(t)')
for t in Q_adapt[:, 0]:
    ax.axvline(x=t, color='grey', linestyle='--', linewidth=0.5)
ax.set_title('x vs. t con líneas verticales (Pasos Adaptativos)')
ax.set_xlabel('Tiempo [años]')
ax.set_ylabel('x [UA]')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
