"""
Proyecto: Problema Gravitacional de los N-Cuerpos
Autores: Daniel Andres Rojas Ordonez, ..., ..., ...
Fecha: 14-04-2025
Descripción:
    Se implementa un programa que resuelve las ecuaciones de movimiento para N partículas
    en interacción gravitacional mutua usando métodos numéricos. Se incluyen tres integradores:
      1. Runge-Kutta de orden 4 (RK4)
      2. Velocity Verlet
      3. Bulirsch-Stoer
    Se leen las condiciones iniciales desde archivos de datos, se efectúan las conversiones de
    unidades necesarias según el caso y se calculan las energías para evaluar la convergencia y
    conservación del sistema. Finalmente se grafican las órbitas y la evolución de la energía.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt, pi
import sys

##########################
# Métodos de integración
##########################

def RK4(ODE, q0, t0, tf, dt=1e-6):
    """
    Método de Runge-Kutta de orden 4 para resolver sistemas de EDO.
    Entradas:
      ODE : función que define el sistema de EDO, debe recibir (t, q) y devolver dq/dt
      q0  : condición inicial (numpy array, forma: [N_p, 6])
      t0  : tiempo inicial
      tf  : tiempo final
      dt  : paso de integración
    Devuelve:
      q   : arreglo de solución de dimensiones [n_steps, N_p, 6]
    """
    n = int((tf - t0) / dt)
    tt = np.linspace(t0, tf, n)
    N = len(q0)
    q = np.zeros([n, N, 6])
    q[0, :, :] = q0

    for i in range(n - 1):
        k1 = dt * ODE(tt[i], q[i, :, :])
        k2 = dt * ODE(tt[i] + dt / 2., q[i, :, :] + k1 / 2.)
        k3 = dt * ODE(tt[i] + dt / 2., q[i, :, :] + k2 / 2.)
        k4 = dt * ODE(tt[i] + dt, q[i, :, :] + k3)
        q[i + 1, :, :] = q[i, :, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.
    return q

def velVerlet(ODE, q0, t0, tf, dt=1e-6):
    """
    Método Velocity-Verlet para resolver sistemas de EDO.
    Entradas similares a RK4.
    """
    n = int((tf - t0) / dt)
    tt = np.linspace(t0, tf, n)
    N = len(q0)
    q = np.zeros([n, N, 6])
    q[0, :, :] = q0

    for i in range(1, n):
        v_half = q[i - 1, :, 3:] + ODE(tt[i - 1], q[i - 1, :, :])[:, 3:] * dt / 2
        q[i, :, 0:3] = q[i - 1, :, 0:3] + v_half * dt
        q[i, :, 3:] = v_half + ODE(tt[i], q[i, :, :])[:, 3:] * dt / 2
    return q

# ============================
# Método Bulirsch-Stoer
# ============================

def modified_midpoint(ODE, t, q, dt, n_steps):
    """
    Método de punto medio modificado con corrección final (Euler final)
    según la formulación:
    
      y0 = q
      y1 = y0 + h * f(t, y0)
      Para i = 1, ..., n_steps-1:
          y_{i+1} = y_{i-1} + 2h * f(t + i*h, y_i)
      y^E = y_{n_steps-1} + h * f(t + dt, y_{n_steps})
      Y_N = 0.5 * (y_{n_steps} + y^E)
      
    Entradas:
      ODE    : función que define el sistema (q' = f(t, q))
      t      : tiempo inicial del subintervalo
      q      : condición inicial (arreglo de dimensión [N,6])
      dt     : intervalo total de integración
      n_steps: número de subpasos usados en el método de punto medio
    Devuelve:
      q_final: aproximación de la solución en t+dt usando el método de punto medio modificado
    """
    h = dt / n_steps
    y0 = q
    y1 = y0 + h * ODE(t, y0)
    y_prev = y0
    y_curr = y1
    for i in range(1, n_steps):
        current_t = t + i * h
        y_next = y_prev + 2 * h * ODE(current_t, y_curr)
        y_prev = y_curr
        y_curr = y_next
    # Realizamos el paso final tipo Euler:
    y_euler = y_prev + h * ODE(t + dt, y_curr)
    # Promediamos la última estimación y el paso Euler
    q_final = 0.5 * (y_curr + y_euler)
    return q_final

def bulirsch_stoer_step(ODE, t, q, dt, tol):
    """
    Realiza un paso de integración con Bulirsch-Stoer para un intervalo dt.
    Se utiliza la extrapolación de Richardson a partir del método de punto medio modificado.
    
    Entradas:
      ODE   : función que define el sistema (q' = f(t,q))
      t     : tiempo inicial
      q     : condición inicial (arreglo de dimensión [N,6])
      dt    : paso de integración
      tol   : tolerancia deseada para el error local
    Devuelve:
      q_extrap: solución extrapolada en t+dt
      error   : estimación del error
    """
    n_seq = [2, 4, 6, 8, 10, 12]
    R = []  # Almacena las soluciones para cada n_steps y sus extrapolaciones
    error = np.inf
    for k, n in enumerate(n_seq):
        y = modified_midpoint(ODE, t, q, dt, n)
        R.append(y)
        # Extrapolación de Richardson de forma recursiva:
        for j in range(k - 1, -1, -1):
            factor = (n_seq[k] / n_seq[j])**2 - 1.0
            R[j] = R[j + 1] + (R[j + 1] - R[j]) / factor
        if k > 0:
            # Se estima el error con la diferencia entre las dos mejores aproximaciones:
            error = np.linalg.norm(R[0] - R[1])
            if error < tol:
                return R[0], error
    return R[0], error

def bulirsch_stoer(ODE, q0, t0, tf, dt=1e-6, tol=1e-8):
    """
    Método Bulirsch-Stoer para resolver sistemas de EDO de forma adaptativa.
    
    Entradas:
      ODE : función que define el sistema (debe recibir (t,q) y devolver dq/dt)
      q0  : condición inicial (arreglo de forma [N,6])
      t0  : tiempo inicial
      tf  : tiempo final
      dt  : tamaño inicial del paso máximo
      tol : tolerancia para el error local
    Devuelve:
      q   : arreglo de solución con dimensiones [n_steps, N, 6],
            donde n_steps es variable según el control del error.
    """
    t = t0
    q_list = [q0]
    t_list = [t0]
    
    while t < tf:
        dt_current = min(dt, tf - t)
        q_new, err = bulirsch_stoer_step(ODE, t, q_list[-1], dt_current, tol)
        # Si el error es aceptable, se acepta el paso y se intenta aumentar dt
        if err < tol:
            t = t + dt_current
            q_list.append(q_new)
            t_list.append(t)
            dt = dt_current * min(2, (tol / err)**0.25)
        else:
            # Si el error es muy grande se reduce dt y se reintenta el paso
            dt = dt_current * max(0.1, (tol / err)**0.25)
    return np.array(q_list)

##########################
# Clase para el sistema
##########################

class System:
    def __init__(self, mass, G=4 * pi**2):
        """
        Inicializa el sistema de N partículas.
          mass: arreglo de masas (todas deben ser positivas)
          G: constante gravitacional (por defecto en unidades AU, años, masas solares)
        """
        if np.any(mass <= 0):
            sys.exit('Todas las masas deben ser positivas')
        self.N = len(mass)
        self.mass = mass
        self.G = G

    def EoM(self, t, q):
        """
        Ecuaciones de movimiento para N partículas bajo interacción gravitacional.
        q: arreglo con [x, y, z, vx, vy, vz] para cada partícula.
        Devuelve dq/dt de forma (N,6)
        """
        dqdt = np.zeros(q.shape)
        # Primeras tres componentes: las velocidades
        dqdt[:, 0:3] = q[:, 3:]
        # Aceleraciones: fuerza gravitatoria entre partículas
        for i in range(self.N):
            Delta = q[i, 0:3] - q[:, 0:3]
            r = np.sqrt(np.sum(Delta**2, axis=1))
            # Para evitar división por cero en el auto-término, se asigna r[i] = 1
            r[i] = 1.0
            dqdt[i, 3] = -self.G * np.sum(Delta[:, 0] * self.mass / r**3)
            dqdt[i, 4] = -self.G * np.sum(Delta[:, 1] * self.mass / r**3)
            dqdt[i, 5] = -self.G * np.sum(Delta[:, 2] * self.mass / r**3)
        return dqdt

    def KineticEnergy(self, q):
        """
        Calcula la energía cinética total.
        """
        v2 = np.sum(q[:, 3:]**2, axis=1)
        T = 0.5 * np.sum(self.mass * v2)
        return T

    def PotentialEnergy(self, q):
        """
        Calcula la energía potencial total.
        """
        (x, y, z, _, _, _) = q.transpose()
        U = 0.0
        for i in range(self.N):
            deltax = x[i] - x
            deltay = y[i] - y
            deltaz = z[i] - z
            r = np.sqrt(deltax**2 + deltay**2 + deltaz**2)
            # Evitar la división por cero: se asigna un valor muy grande al término i=j
            r[i] = 1e300
            U += -0.5 * self.G * self.mass[i] * np.sum(self.mass / r)
        return U

    def TotalEnergy(self, q):
        """
        Retorna la energía total (cinética + potencial).
        """
        return self.KineticEnergy(q) + self.PotentialEnergy(q)


##########################
# Rutina para calcular la evolución de la energía
##########################

def compute_energy_evolution(system, q_sol):
    """
    Calcula la energía cinética, potencial y total del sistema en cada paso.
    
    Entradas:
      system: objeto de la clase System, que proporciona los métodos KineticEnergy y PotentialEnergy.
      q_sol : arreglo de solución de dimensiones [n_steps, N, 6]
    Devuelve:
      T: arreglo con la energía cinética en cada paso.
      U: arreglo con la energía potencial en cada paso.
      E: arreglo con la energía total en cada paso.
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
# Funciones de graficación
##########################

def plot3D(q, names, integrator='', savefig=False, filename='orbit.png'):
    """
    Grafica las trayectorias 3D de cada partícula.
      q: arreglo de dimensiones [n_steps, N, 6]
      names: lista de listas [nombre, color]
      integrator: nombre del método numérico usado (cadena)
    """
    boundary = max(abs(np.amax(q[:, :, 0:3])), abs(np.amin(q[:, :, 0:3])))*(1 + 0.1)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    for i in range(len(names)):
        ax.plot(q[:, i, 0], q[:, i, 1], q[:, i, 2], label=names[i][0], color=names[i][1])
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_zlabel('z (AU)')
    ax.set_xlim(-boundary, boundary)
    ax.set_ylim(-boundary, boundary)
    ax.set_zlim(-boundary, boundary)
    ax.legend()
    ax.set_title('Órbitas calculadas usando ' + integrator)
    if savefig:
        plt.savefig(filename)
    plt.show()

def energyPlot(T, U, E, integrator='', savefig=False, filename='energy.png'):
    """
    Grafica la evolución de las energías cinética, potencial y total.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot()
    ax.plot(T, color='cornflowerblue', label='Energía Cinética')
    ax.plot(U, color='crimson', label='Energía Potencial')
    ax.plot(E, color='black', label='Energía Total')
    ax.set_xlabel('Tiempo (pasos de integración)')
    ax.set_ylabel('Energía')
    ax.set_title('Evolución Energética usando ' + integrator)
    ax.grid(True)
    ax.legend()
    if savefig:
        plt.savefig(filename)
    plt.show()

##########################
# Funciones para leer datos
##########################

def read_data(filename, system_type="sun_earth"):
    """
    Lee el archivo de datos y realiza las conversiones de unidades según el sistema.
    Parámetros:
      filename    : nombre del archivo a leer.
      system_type : "sun_earth" o "S0stars"
    Retorna:
      x, y, z, vx, vy, vz, mass (arreglos numpy)
    """
    data = np.loadtxt(filename)
    if system_type == "sun_earth":
        AU = 1.49598e11         # metros en 1 AU
        year = 3.15576e7        # segundos en 1 año
        mass_sun = 1.98855e30   # kg en 1 masa solar
        pos_factor = 1.0 / AU
        vel_factor = year / AU  # (m/s) -> (AU/año)
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
# Función principal
##########################

def main():
    print("Seleccione el sistema a simular:")
    print("  1. Sistema Sol-Tierra")
    print("  2. Sistema S0 (13 estrellas + SgrA*)")
    opcion = input("Opción (1/2): ").strip()
    steps_n = int(input("Número de pasos a integrar: "))
    print("Número de pasos a integrar: ", steps_n)
    Anios_simulacion = float(input("Años de simulación: "))
    print("Años a simular: ", Anios_simulacion)

    if opcion == "1":
        system_type = "sun_earth"
        filename = "sun_earth.dat"
        names = [['Sol', 'gold'], ['Tierra', 'blue']]
        t0 = 0.0
        tf = Anios_simulacion    # 2 años de simulación aproximadamente
        steps = steps_n   # Para métodos de paso fijo (no se usa en Bulirsch-Stoer)
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
        print("Opción no reconocida. Saliendo ...")
        return

    x, y, z, vx, vy, vz, mass = read_data(filename, system_type)
    N = len(mass)
    print("Número de partículas =", N)
    
    q0 = np.array([x, y, z, vx, vy, vz]).T
    t0_val = t0
    tf_val = tf
    dt = (tf_val - t0_val) / steps

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
        print("Opción de integrador no válida. Se usará RK4 por defecto.")
        integrator_used = "RK4"

    G = 4 * pi**2
    S = System(mass, G)

    print("\nRealizando la integración con", integrator_used, "...")
    start_time = time.time()
    if integrator_used == "RK4":
        q = RK4(S.EoM, q0, t0_val, tf_val, dt)
    elif integrator_used == "Velocity Verlet":
        q = velVerlet(S.EoM, q0, t0_val, tf_val, dt)
    elif integrator_used == "Bulirsch-Stoer":
        tol = 1e-8
        q = bulirsch_stoer(S.EoM, q0, t0_val, tf_val, dt, tol)
    end_time = time.time()
    print("\nTiempo de cómputo usando", integrator_used, "fue:", end_time - start_time, "segundos.\n")
    
    # Para métodos adaptativos (Bulirsch-Stoer), se toma el número real de pasos en la solución:
    n_steps = q.shape[0]
    
    # Calcular la evolución de la energía total usando la rutina incorporada.
    T, U, E = compute_energy_evolution(S, q)
    energia_inicial = E[0]
    energia_final = E[-1]
    print("Cambio en la energía total (E_inicial - E_final):", energia_inicial - energia_final, "\n")
    
    # Graficar las órbitas (se muestrea la solución para acelerar la visualización)
    stride = max(1, n_steps // 100)
    plot3D(q[::stride], names, integrator_used)
    
    # Graficar la evolución energética
    energyPlot(T, U, E, integrator_used)
    
if __name__ == "__main__":
    main()
