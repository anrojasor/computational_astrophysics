import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from math import pi
import sys

##########################
# MÉTODOS DE INTEGRACIÓN
##########################

def RK4(ODE, q0, t0, tf, dt=1e-6):
    """
    Integra un sistema de EDO usando el método de Runge-Kutta de 4º orden.

    Parámetros:
      ODE: función que define el sistema, recibe (t, q) y devuelve dq/dt.
      q0 : condición inicial (array de forma [N,6]).
      t0 : tiempo inicial.
      tf : tiempo final.
      dt : paso de integración.
      
    Retorna:
      q  : array de solución de forma [n_steps, N, 6].
    """
    n = int((tf - t0) / dt)
    N = len(q0)
    q = np.zeros([n, N, 6])
    q[0, :, :] = q0

    for i in range(n - 1):
        k1 = dt * ODE(t0 + i * dt, q[i, :, :])
        k2 = dt * ODE(t0 + i * dt + dt / 2., q[i, :, :] + k1 / 2.)
        k3 = dt * ODE(t0 + i * dt + dt / 2., q[i, :, :] + k2 / 2.)
        k4 = dt * ODE(t0 + i * dt + dt, q[i, :, :] + k3)
        q[i + 1, :, :] = q[i, :, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.
    return q

def velVerlet(ODE, q0, t0, tf, dt=1e-6):
    """
    Integra un sistema de EDO usando el método Velocity-Verlet.

    Parámetros similares a RK4.
      
    Retorna:
      q  : array de solución de forma [n_steps, N, 6].
    """
    n = int((tf - t0) / dt)
    N = len(q0)
    q = np.zeros([n, N, 6])
    q[0, :, :] = q0

    for i in range(1, n):
        v_half = q[i - 1, :, 3:] + ODE(t0 + (i - 1) * dt, q[i - 1, :, :])[:, 3:] * dt / 2
        q[i, :, 0:3] = q[i - 1, :, 0:3] + v_half * dt
        q[i, :, 3:] = v_half + ODE(t0 + i * dt, q[i, :, :])[:, 3:] * dt / 2
    return q

def modified_midpoint(ODE, t, q, dt, n_steps):
    """
    Método de punto medio modificado con corrección final mediante paso Euler.
    
    Se ejecuta:
      y0 = q
      y1 = y0 + h * f(t, y0)
      y_{i+1} = y_{i-1} + 2h * f(t + i*h, y_i)  para i=1,...,n_steps-1
      y^E = y_{n_steps-1} + h * f(t+dt, y_{n_steps})
      Y_N = 0.5 * (y_{n_steps} + y^E)
      
    Parámetros:
      ODE   : función del sistema.
      t     : tiempo inicial del subintervalo.
      q     : condición inicial (array de forma [N,6]).
      dt    : intervalo total de integración.
      n_steps: número de subpasos.
      
    Retorna:
      q_final: aproximación de la solución en t+dt.
    """
    h = dt / n_steps
    y0 = q
    y1 = y0 + h * ODE(t, y0)
    y_prev, y_curr = y0, y1
    for i in range(1, n_steps):
        current_t = t + i * h
        y_next = y_prev + 2 * h * ODE(current_t, y_curr)
        y_prev, y_curr = y_curr, y_next
    y_euler = y_prev + h * ODE(t + dt, y_curr)
    q_final = 0.5 * (y_curr + y_euler)
    return q_final

def bulirsch_stoer_step(ODE, t, q, dt, tol):
    """
    Realiza un paso de integración Bulirsch-Stoer para un intervalo dt
    utilizando extrapolación de Richardson.
    
    Parámetros:
      ODE  : función del sistema.
      t    : tiempo inicial.
      q    : condición inicial (array [N,6]).
      dt   : paso de integración.
      tol  : tolerancia para el error.
      
    Retorna:
      q_extrap: solución extrapolada en t+dt.
      error   : error estimado.
    """
    n_seq = [2, 4, 6, 8, 10, 12]
    R = []
    for k, n in enumerate(n_seq):
        y = modified_midpoint(ODE, t, q, dt, n)
        R.append(y)
        for j in range(k - 1, -1, -1):
            factor = (n_seq[k] / n_seq[j]) ** 2 - 1.0
            R[j] = R[j + 1] + (R[j + 1] - R[j]) / factor
        if k > 0:
            error = np.linalg.norm(R[0] - R[1])
            if error < tol:
                return R[0], error
    return R[0], error

def bulirsch_stoer(ODE, q0, t0, tf, dt=1e-6, tol=1e-8):
    """
    Integra un sistema de EDO de forma adaptativa usando el método Bulirsch-Stoer.
    
    Parámetros:
      ODE : función del sistema.
      q0  : condición inicial (array [N,6]).
      t0  : tiempo inicial.
      tf  : tiempo final.
      dt  : tamaño inicial del paso.
      tol : tolerancia para el error local.
      
    Retorna:
      q : array de solución con dimensiones [n_steps, N, 6].
    """
    t = t0
    q_list = [q0]
    
    while t < tf:
        dt_current = min(dt, tf - t)
        q_new, err = bulirsch_stoer_step(ODE, t, q_list[-1], dt_current, tol)
        if err < tol:
            t += dt_current
            q_list.append(q_new)
            dt = dt_current * min(2, (tol / err) ** 0.25)
        else:
            dt = dt_current * max(0.1, (tol / err) ** 0.25)
    return np.array(q_list)

##########################
# CLASE DEL SISTEMA GRAVITACIONAL
##########################

class System:
    def __init__(self, mass, G=4 * pi**2):
        """
        Inicializa el sistema de N partículas.
        
        Parámetros:
          mass: array de masas (en masas solares; deben ser positivas).
          G   : constante gravitacional en [AU^3 / (yr^2 · M☉)] (por defecto 4π²).
        """
        if np.any(mass <= 0):
            sys.exit('Todas las masas deben ser positivas')
        self.N = len(mass)
        self.mass = mass
        self.G = G

    def EoM(self, t, q):
        """
        Retorna las ecuaciones de movimiento para N partículas en [x, y, z, vx, vy, vz].
        """
        dqdt = np.zeros(q.shape)
        dqdt[:, 0:3] = q[:, 3:]
        for i in range(self.N):
            Delta = q[i, 0:3] - q[:, 0:3]
            r = np.sqrt(np.sum(Delta**2, axis=1))
            r[i] = 1.0  # Evitar división por cero en el autointeracción.
            dqdt[i, 3] = -self.G * np.sum(Delta[:, 0] * self.mass / r**3)
            dqdt[i, 4] = -self.G * np.sum(Delta[:, 1] * self.mass / r**3)
            dqdt[i, 5] = -self.G * np.sum(Delta[:, 2] * self.mass / r**3)
        return dqdt

    def KineticEnergy(self, q):
        """Calcula la energía cinética total."""
        v2 = np.sum(q[:, 3:] ** 2, axis=1)
        return 0.5 * np.sum(self.mass * v2)

    def PotentialEnergy(self, q):
        """Calcula la energía potencial total (evitando doble-cuenta)."""
        (x, y, z, _, _, _) = q.transpose()
        U = 0.0
        for i in range(self.N):
            deltax = x[i] - x
            deltay = y[i] - y
            deltaz = z[i] - z
            r = np.sqrt(deltax**2 + deltay**2 + deltaz**2)
            r[i] = 1e300  # Ignorar la autointeracción
            U += -0.5 * self.G * self.mass[i] * np.sum(self.mass / r)
        return U

    def TotalEnergy(self, q):
        """Retorna la energía total (cinética + potencial)."""
        return self.KineticEnergy(q) + self.PotentialEnergy(q)

##########################
# EVOLUCIÓN DE LA ENERGÍA
##########################

def compute_energy_evolution(system, q_sol):
    """
    Calcula la energía cinética, potencial y total en cada paso.
    
    Parámetros:
      system: instancia de System.
      q_sol : array solución [n_steps, N, 6].
      
    Retorna:
      T, U, E : arrays de energía cinética, potencial y total.
    """
    n_steps = q_sol.shape[0]
    T = np.zeros(n_steps)
    U = np.zeros(n_steps)
    E = np.zeros(n_steps)
    for i in range(n_steps):
        T[i] = system.KineticEnergy(q_sol[i, :, :])
        U[i] = system.PotentialEnergy(q_sol[i, :, :])
        E[i] = T[i] + U[i]
    return T, U, E

##########################
# GRAFICACIÓN
##########################

def plot3D(q, names, integrator='', savefig=False, filename='orbit.png'):
    """
    Grafica las trayectorias 3D de cada partícula en unidades de AU.
    """
    boundary = max(abs(np.amax(q[:, :, 0:3])), abs(np.amin(q[:, :, 0:3]))) * 1.1
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(names)):
        ax.plot(q[:, i, 0], q[:, i, 1], q[:, i, 2],
                label=names[i][0], color=names[i][1])
    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_zlabel('z [AU]')
    ax.set_xlim(-boundary, boundary)
    ax.set_ylim(-boundary, boundary)
    ax.set_zlim(-boundary, boundary)
    ax.legend()
    ax.set_title(f'Órbitas calculadas usando {integrator}')
    if savefig:
        plt.savefig(filename)
    plt.show()

def energyPlot(T, U, E, integrator='', savefig=False, filename='energy.png'):
    """
    Grafica la evolución de la energía cinética, potencial y total.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot()
    ax.plot(T, color='cornflowerblue', label='Energía Cinética')
    ax.plot(U, color='crimson', label='Energía Potencial')
    ax.plot(E, color='black', label='Energía Total')
    ax.set_xlabel('Tiempo (pasos)')
    ax.set_ylabel('Energía')
    ax.set_title(f'Evolución Energética usando {integrator}')
    ax.grid(True)
    ax.legend()
    if savefig:
        plt.savefig(filename)
    plt.show()

##########################
# LECTURA DE DATOS
##########################

def read_data(filename, system_type="sun_earth"):
    """
    Lee un archivo de datos y convierte las unidades a AU, años y masas solares.
    """
    data = np.loadtxt(filename)
    if system_type == "sun_earth":
        AU = 1.49598e11         # m en 1 AU
        year = 3.15576e7        # s en 1 año
        mass_sun = 1.98855e30   # kg en 1 masa solar
        pos_factor = 1.0 / AU
        vel_factor = year / AU  # de m/s a AU/yr
        mass_factor = 1.0 / mass_sun
        x = data[:, 0] * pos_factor
        y = data[:, 1] * pos_factor
        z = data[:, 2] * pos_factor
        vx = data[:, 3] * vel_factor
        vy = data[:, 4] * vel_factor
        vz = data[:, 5] * vel_factor
        mass = data[:, 6] * mass_factor
    elif system_type == "S0stars":
        arcsec_in_au = 8000.0
        x = data[:, 0] * arcsec_in_au
        y = data[:, 1] * arcsec_in_au
        z = data[:, 2] * arcsec_in_au
        vx = data[:, 3] * arcsec_in_au
        vy = data[:, 4] * arcsec_in_au
        vz = data[:, 5] * arcsec_in_au
        mass = data[:, 6]
    else:
        sys.exit("Tipo de sistema desconocido.")
    return x, y, z, vx, vy, vz, mass

##########################
# ANIMACIÓN 3D DE ÓRBITAS
##########################

def animate_orbit_3d(q, dt, names, title='Animación 3D de órbitas', interval=50):
    """
    Crea una animación 3D de la evolución de las órbitas en AU, mostrando el tiempo en años.
    
    Parámetros:
      q     : array de solución [n_steps, N, 6] (posición en columnas 0,1,2 en AU).
      dt    : paso de tiempo en años.
      names : lista de [nombre, color] para cada cuerpo.
      title : título de la animación.
      interval: intervalo entre cuadros (ms).
      
    Retorna:
      ani   : objeto FuncAnimation.
    """
    from mpl_toolkits.mplot3d import Axes3D
    n_steps, N, _ = q.shape

    x_all = q[:, :, 0]
    y_all = q[:, :, 1]
    z_all = q[:, :, 2]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(np.min(x_all) * 1.1, np.max(x_all) * 1.1)
    ax.set_ylim(np.min(y_all) * 1.1, np.max(y_all) * 1.1)
    ax.set_zlim(np.min(z_all) * 1.1, np.max(z_all) * 1.1)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")
    ax.set_title(title)

    lines, points = [], []
    for i in range(N):
        line, = ax.plot([], [], [], lw=2, label=names[i][0], color=names[i][1])
        point, = ax.plot([], [], [], 'o', markersize=8, color=names[i][1])
        lines.append(line)
        points.append(point)

    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        time_text.set_text("")
        return lines + points + [time_text]

    def animate(i):
        for j in range(N):
            x = q[:i, j, 0]
            y = q[:i, j, 1]
            z = q[:i, j, 2]
            lines[j].set_data(x, y)
            lines[j].set_3d_properties(z)
            points[j].set_data([q[i, j, 0]], [q[i, j, 1]])
            points[j].set_3d_properties([q[i, j, 2]])
        current_years = i * dt
        time_text.set_text(f"Tiempo = {current_years:.2f} años")
        return lines + points + [time_text]

    ani = animation.FuncAnimation(fig, animate, frames=n_steps,
                                  init_func=init, interval=interval, blit=True)
    plt.show()
    return ani

##########################
# FUNCIÓN PRINCIPAL
##########################

def main():
    # Selección del sistema y parámetros de simulación
    print("Seleccione el sistema a simular:")
    print("  1. Sistema Sol-Tierra")
    print("  2. Sistema S0 (13 estrellas + SgrA*)")
    opcion = input("Opción (1/2): ").strip()
    steps_n = int(input("Número de pasos a integrar: "))
    Anios_simulacion = float(input("Años de simulación: "))
    
    if opcion == "1":
        system_type = "sun_earth"
        filename = "sun_earth.dat"
        names = [['Sol', 'gold'], ['Tierra', 'blue']]
        t0 = 0.0
        tf = Anios_simulacion  # tiempo en años
        steps = steps_n
        print("\nSimulando el sistema Sol-Tierra...")
    elif opcion == "2":
        system_type = "S0stars"
        filename = "S0stars.dat"
        names = [
            ['SgrA*', 'black'],
            ['Star01', 'crimson'],
            ['Star02', 'cornflowerblue'],
            ['Star03', 'darkgreen'],
            ['Star04', 'darkorange'],
            ['Star05', 'darkviolet'],
            ['Star06', 'darkturquoise'],
            ['Star07', 'deeppink'],
            ['Star08', 'gold'],
            ['Star09', 'indigo'],
            ['Star10', 'lime'],
            ['Star11', 'maroon'],
            ['Star12', 'navy'],
            ['Star13', 'olive']
        ]
        t0 = 0.0
        tf = Anios_simulacion
        steps = steps_n
        print("\nSimulando el sistema S0 (13 estrellas + SgrA*)...")
    else:
        sys.exit("Opción no reconocida.")

    x, y, z, vx, vy, vz, mass = read_data(filename, system_type)
    N = len(mass)
    print("Número de partículas =", N)
    
    q0 = np.array([x, y, z, vx, vy, vz]).T  # Estado inicial [N,6]
    t0_val = t0
    tf_val = tf
    dt = (tf_val - t0_val) / steps  # dt en años

    # Selección del integrador
    print("\nSeleccione el integrador:")
    print("  1. RK4")
    print("  2. Velocity Verlet")
    print("  3. Bulirsch-Stoer")
    op_int = input("Opción (1/2/3): ").strip()
    if op_int == "1":
        integrator_used = "RK4"
    elif op_int == "2":
        integrator_used = "Velocity Verlet"
    elif op_int == "3":
        integrator_used = "Bulirsch-Stoer"
    else:
        integrator_used = "RK4"
        print("Integrador inválido. Se usará RK4 por defecto.")

    G = 4 * pi**2
    S = System(mass, G)

    print(f"\nRealizando la integración con {integrator_used}...")
    start_time = time.time()
    if integrator_used == "RK4":
        q = RK4(S.EoM, q0, t0_val, tf_val, dt)
    elif integrator_used == "Velocity Verlet":
        q = velVerlet(S.EoM, q0, t0_val, tf_val, dt)
    elif integrator_used == "Bulirsch-Stoer":
        tol = 1e-8
        q = bulirsch_stoer(S.EoM, q0, t0_val, tf_val, dt, tol)
    end_time = time.time()
    print(f"\nTiempo de cómputo usando {integrator_used} fue: {end_time - start_time:.2f} segundos.\n")
    
    n_steps = q.shape[0]
    T, U, E = compute_energy_evolution(S, q)
    print(f"Cambio en la energía total (inicial - final): {E[0] - E[-1]:.2e}\n")
    
    energyPlot(T, U, E, integrator_used)

    # Graficar la trayectoria en 3D para integradores de paso fijo
    if integrator_used in ["RK4", "Velocity Verlet"]:
        plot3D(q[::max(1, n_steps // 100)], names, integrator_used)
    
    print("\nGenerando animación 3D de las órbitas...")
    animate_orbit_3d(q, dt, names, title=f"Animación 3D ({integrator_used})", interval=20)

if __name__ == "__main__":
    main()
