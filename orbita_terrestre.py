import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. DATOS Y CONSTANTES
# =============================================================================

# Constante de gravitación universal [m^3 / (kg·s^2)]
G = 6.67430e-11  

# Masa del Sol [kg]
M_sun = 1.98847e30  

# =============================================================================
# 2. FUNCIONES BÁSICAS: ACELERACIÓN Y ENERGÍAS
# =============================================================================

def aceleracion(x, y):
    """
    Retorna la aceleración (ax, ay) debida a la gravedad del Sol
    para una partícula en la posición (x, y).
    Ecuación: a = -G M_sun / r^3 * (x, y), con r = sqrt(x^2 + y^2).
    """
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    factor = -G * M_sun / (r**3)
    ax = factor * x
    ay = factor * y
    return ax, ay

def energias(x, y, vx, vy):
    """
    Calcula la energía potencial, cinética y total (por unidad de masa),
    asumiendo masa de la partícula = m, pero usamos energía por unidad de masa (E/m):
      U(r)/m = - G M_sun / r
      K(v)/m = 0.5 * (vx^2 + vy^2)
      E = U + K
    """
    r = np.sqrt(x**2 + y**2)
    U = -G * M_sun / r
    K = 0.5 * (vx**2 + vy**2)
    return U, K, U + K

# =============================================================================
# 3. MÉTODO SIMPLÉCTICO (VELOCITY-VERLET) Y RK4 EN 2D
# =============================================================================

def velocity_verlet(x0, y0, vx0, vy0, dt, N):
    """
    Implementa el método Velocity-Verlet (simpéctico) en 2D.
    Retorna arrays con x, y, vx, vy en cada paso.
    """
    xs = np.zeros(N)
    ys = np.zeros(N)
    vxs = np.zeros(N)
    vys = np.zeros(N)

    # Condiciones iniciales
    xs[0] = x0
    ys[0] = y0
    vxs[0] = vx0
    vys[0] = vy0

    # Aceleración inicial
    ax0, ay0 = aceleracion(x0, y0)

    for i in range(N-1):
        # Actualizar posición
        xs[i+1] = xs[i] + vxs[i]*dt + 0.5*ax0*(dt**2)
        ys[i+1] = ys[i] + vys[i]*dt + 0.5*ay0*(dt**2)

        # Calcular nueva aceleración
        ax1, ay1 = aceleracion(xs[i+1], ys[i+1])

        # Actualizar velocidad
        vxs[i+1] = vxs[i] + 0.5*(ax0 + ax1)*dt
        vys[i+1] = vys[i] + 0.5*(ay0 + ay1)*dt

        # Para el siguiente ciclo
        ax0, ay0 = ax1, ay1

    return xs, ys, vxs, vys

def rk4_2d(x0, y0, vx0, vy0, dt, N):
    """
    Implementa el método de Runge-Kutta de 4º orden para el movimiento 2D.
    Retorna arrays con x, y, vx, vy en cada paso.
    """
    xs = np.zeros(N)
    ys = np.zeros(N)
    vxs = np.zeros(N)
    vys = np.zeros(N)

    xs[0] = x0
    ys[0] = y0
    vxs[0] = vx0
    vys[0] = vy0

    for i in range(N-1):
        # k1
        ax1, ay1 = aceleracion(xs[i], ys[i])
        k1x = vxs[i]
        k1y = vys[i]
        k1vx = ax1
        k1vy = ay1

        # k2
        ax2, ay2 = aceleracion(xs[i] + 0.5*dt*k1x, ys[i] + 0.5*dt*k1y)
        k2x = vxs[i] + 0.5*dt*k1vx
        k2y = vys[i] + 0.5*dt*k1vy
        k2vx = ax2
        k2vy = ay2

        # k3
        ax3, ay3 = aceleracion(xs[i] + 0.5*dt*k2x, ys[i] + 0.5*dt*k2y)
        k3x = vxs[i] + 0.5*dt*k2vx
        k3y = vys[i] + 0.5*dt*k2vy
        k3vx = ax3
        k3vy = ay3

        # k4
        ax4, ay4 = aceleracion(xs[i] + dt*k3x, ys[i] + dt*k3y)
        k4x = vxs[i] + dt*k3vx
        k4y = vys[i] + dt*k3vy
        k4vx = ax4
        k4vy = ay4

        # Actualizar
        xs[i+1] = xs[i] + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        ys[i+1] = ys[i] + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        vxs[i+1] = vxs[i] + (dt/6.0)*(k1vx + 2*k2vx + 2*k3vx + k4vx)
        vys[i+1] = vys[i] + (dt/6.0)*(k1vy + 2*k2vy + 2*k3vy + k4vy)

    return xs, ys, vxs, vys

# =============================================================================
# 4. ORBITA DE LA TIERRA (EJEMPLO DE CONTROL)
# =============================================================================

# Datos para la Tierra en perihelio:
r_perihelio_tierra = 1.4710e11    # m
v_perihelio_tierra = 3.0287e4     # m/s

# Condiciones iniciales (2D) para la Tierra
x0_earth = r_perihelio_tierra
y0_earth = 0.0
vx0_earth = 0.0
vy0_earth = v_perihelio_tierra

# Tiempo total ~ 1 año en segundos
T_earth = 3.154e7
# Paso temporal (ej. 1 hora)
dt_earth = 3600.0
N_earth = int(T_earth/dt_earth) + 1

print("Simulando órbita terrestre (~1 año)...")
xs_vv_e, ys_vv_e, vxs_vv_e, vys_vv_e = velocity_verlet(x0_earth, y0_earth, vx0_earth, vy0_earth, dt_earth, N_earth)

# Graficamos la órbita de la Tierra (sólo con simpéctico como demo):
plt.figure(figsize=(6,6))
plt.plot(xs_vv_e, ys_vv_e, label='Tierra (Verlet)', color='blue')
plt.plot([0],[0], marker='*', markersize=15, color='orange', label='Sol')
plt.title('Órbita Terrestre en el plano XY')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# 5. NUEVA ORBITA (VARIACIÓN DE LA VELOCIDAD)
# =============================================================================

# Ejemplo: incrementar la velocidad del perihelio en un 5% para la Tierra
vy0_variada = 1.05 * v_perihelio_tierra
xs_vv_e2, ys_vv_e2, vxs_vv_e2, vys_vv_e2 = velocity_verlet(x0_earth, y0_earth, vx0_earth, vy0_variada, dt_earth, N_earth)

plt.figure(figsize=(6,6))
plt.plot(xs_vv_e, ys_vv_e, label='Órbita Original', color='blue')
plt.plot(xs_vv_e2, ys_vv_e2, label='Órbita con v+5%', color='green')
plt.plot([0],[0], marker='*', markersize=15, color='orange', label='Sol')
plt.title('Comparación de órbitas terrestres con distinta velocidad inicial')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# 6. ORBITA DE PLUTÓN
# =============================================================================

# 4.4386e12 m en el perihelio, 6.1218e3 m/s de velocidad
r_perihelio_pluto = 4.4386e12
v_perihelio_pluto = 6.1218e3

# Condiciones iniciales (2D) para Plutón en su perihelio
x0_pluto = r_perihelio_pluto
y0_pluto = 0.0
vx0_pluto = 0.0
vy0_pluto = v_perihelio_pluto

# Periodo orbital aproximado de Plutón ~ 248 años
# 1 año ~ 3.154e7 s  =>  T_pluto ~ 248 * 3.154e7 ~ 7.82e9 s
T_pluto = 7.82e9

# Paso temporal (por ejemplo, 1 día = 86400 s) para no exagerar en número de pasos
dt_pluto = 86400.0
N_pluto = int(T_pluto / dt_pluto) + 1

print("\nSimulando órbita de Plutón (~1 órbita completa, 248 años)...")
print(f"Número de pasos = {N_pluto} (puede tardar en computadoras lentas).")

# Integramos con el método simpéctico (Velocity-Verlet) para Plutón
xs_vv_p, ys_vv_p, vxs_vv_p, vys_vv_p = velocity_verlet(x0_pluto, y0_pluto, vx0_pluto, vy0_pluto, dt_pluto, N_pluto)

# Graficamos la trayectoria en el plano XY
plt.figure(figsize=(6,6))
plt.plot(xs_vv_p, ys_vv_p, label='Órbita de Plutón (Verlet)', color='purple')
plt.plot([0],[0], marker='*', markersize=15, color='orange', label='Sol')
plt.title('Órbita de Plutón en el plano XY (perihelio)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Si deseas (opcional) graficar también las energías de Plutón:
U_p = np.zeros(N_pluto)
K_p = np.zeros(N_pluto)
E_p = np.zeros(N_pluto)
for i in range(N_pluto):
    U_p[i], K_p[i], E_p[i] = energias(xs_vv_p[i], ys_vv_p[i], vxs_vv_p[i], vys_vv_p[i])

plt.figure(figsize=(10,5))
plt.plot(U_p, label='Energía Potencial (U)', color='blue')
plt.plot(K_p, label='Energía Cinética (K)', color='red')
plt.plot(E_p, label='Energía Total (E=U+K)', color='green')
plt.title('Energías vs. Paso de tiempo - Órbita de Plutón (Velocity-Verlet)')
plt.xlabel('Paso de tiempo')
plt.ylabel('Energía por unidad de masa [J/kg]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

