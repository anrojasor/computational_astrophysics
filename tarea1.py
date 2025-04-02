import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
import math

# ============================================================================
#                                PARÁMETROS 

# ============================================================================
RT = 6378.1363            # Radio de la Tierra
GM = 398600.4405          # km^3/s^2
a  = 1.30262 * RT         # km, semieje mayor
e  = 0.16561              # excentricidad
omega_grad = 15.0          # argumento del pericentro (en grados)
omega = np.deg2rad(omega_grad)  # omega a radianes

tp = Time("2025-03-31 00:00:00", format="iso", scale="utc")  # tiempo de paso por el pericentro

# ============================================================================

#                               FUNCIONES
# ============================================================================

# Resulver la ecuación de kepler (Método de Newton-raphson)
def kepler(l, e, tol=1e-14, max_iter=200):
    """
    Resuelve la ecuación de Kepler: E - e*sin(E) = l por el método de Newton-Raphson.
    Parámetros:
        l        : anomalía media
        e        : excentricidad
        tol      : tolerancia para convergencia
        max_iter : iteraciones máximas
    Retorna:
        E        : anomalía excentrica

    Calculando la derivada de la función f(E) = E - e*sin(E) - l, se obtiene:
        f'(E) = 1 - e*cos(E)
    La ecuación de Newton-Raphson es:
        E_(n+1) = E_n - f(E_n)/f'(E_n)
    Donde:
        E_n es la aproximación actual de E y E_(n+1) es la nueva aproximación.
        f(E_n) es el valor de la función en E_n y f'(E_n) es la derivada de la función en E_n.

    La convergencia se logra cuando el cociente de f y f' es menor que la tolerancia especificada.

    Si la derivada es cero, se detiene el proceso.

    Si el número máximo de iteraciones se alcanza sin convergencia, se detiene el proceso.

    Se devuelve el valor de E que se ha calculado.
    """
    # Valor inicial: si e es pequeña, E0 ~ l; de lo contrario, un valor genérico (pi).
    #E = l if e < 0.8 else math.pi
    E = l
    for i in range(max_iter):
        f  = E - e * np.sin(E) - l
        fp = 1 - e * np.cos(E)
        if abs(fp) < 1e-15:
            break
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


# POSITION(T)
def position(t):
    """
    Retorna (r, phi) en el instante t:
      r   = distancia radial en km
      phi = ángulo en radianes (0 a 2π)

    Hace uso de la función kepler para resolver la ecuación de Kepler y calcular la posición del satélite en función del tiempo.
    Parámetros:
        t : Tiempo en formato astropy Time.
    Retorna:
        r   : distancia radial en km
        phi : ángulo en radianes (0 a 2π)
    
    La función calcula la anomalía media a partir del tiempo transcurrido desde el paso por el pericentro, y luego utiliza la ecuación de Kepler para encontrar la anomalía excéntrica. Finalmente, calcula la anomalía verdadera y ajusta el ángulo a un rango de 0 a 2π.

    La anomalía verdadera se calcula a partir de la anomalía excéntrica utilizando la fórmula:
    f = 2 * arctan( sqrt(1+e) * sin(E/2) / sqrt(1-e) * cos(E/2) ).

    La distancia radial se calcula utilizando la fórmula r = a*(1 - e^2) / (1 + e*cos(f)), donde f es la anomalía verdadera.
    
    La posición angular se calcula como phi = f + omega (mod 2π), donde omega es el argumento del pericentro.

    La función devuelve la distancia radial y el ángulo en radianes.

    La función utiliza la librería astropy para manejar el tiempo y las unidades, y numpy para realizar cálculos matemáticos.
    """
    # Calculo de la anomalía media
    dt = (t - tp).to_value('s')  # Tiempo transcurrido desde el paso por el pericentro en segundos
    l = np.sqrt(GM / a**3) * dt  # Anomalía media

    # Resolver E a partir de la ecuación de Kepler
    E = kepler(l, e)  # Anomalía excéntrica

    # Calcular anomalía verdadera
    # f = 2 * arctan 2( sqrt(1+e) * sin(E/2) / sqrt(1-e) * cos(E/2)
    sin_E_2 = np.sin(E/2)
    cos_E_2 = np.cos(E/2)
    num = np.sqrt(1 + e) * sin_E_2
    den = np.sqrt(1 - e) * cos_E_2
    f = 2.0 * np.arctan2(num, den)  # f en [0, 2π)

    # Ajustar f a [0, 2π)
    f = f % (2.0 * np.pi)

    # r = a*(1 - e^2) / (1 + e*cos(f))
    r = a * (1 - e**2) / (1 + e * np.cos(f))

    # phi = f + omega (mod 2π)
    phi = (f + omega) % (2.0 * np.pi)

    return r, phi

# ORBIT()
def orbit():
    """
    Grafica la órbita del satélite a lo largo de un período.
    """
    # Número de puntos
    n_points = 1000

    # Período orbital
    T = 2 * np.pi * np.sqrt(a**3 / GM)  # Período orbital en segundos
    dt = np.linspace(0, T, n_points)  # Tiempo en segundos
    tiempos = [tp + TimeDelta(d, format='sec') for d in dt]  # Lista de tiempos en formato astropy Time

    # Calcular posiciones
    r_vals = []
    phi_vals = []
    for t in tiempos:
        r, phi = position(t)
        r_vals.append(r)
        phi_vals.append(phi)

    r_vals = np.array(r_vals)
    phi_vals = np.array(phi_vals)

    # Convertir a coordenadas cartesianas
    x_vals = r_vals * np.cos(phi_vals)
    y_vals = r_vals * np.sin(phi_vals)

    # Graficar la órbita
    plt.figure(figsize=(8, 8))
    plt.plot(x_vals, y_vals, label='Órbita del satélite')
    plt.plot(0, 0, 'ro', label='Tierra')  # Tierra en el origen
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title('Órbita del satélite')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()

# DATE()
def date(r0):
    """
    Dado r0 (km), encuentra el instante t0 en el intervalo [tp, tp+T]
    en que r(t0) = r0, con una tolerancia 'tol' en km.
    Retorna un objeto astropy.time.Time (UTC) o None si no lo halla.
    Parámetros:
        r0 : distancia radial en km
    Retorna:
        t0 : instante en que r(t0) = r0 (en formato astropy Time)

    Si no se encuentra un instante t0 que cumpla con la condición, retorna None.

    La función utiliza el método de bisección para encontrar la raíz de la función g(t) = r(t) - r0.

    La función g(t) se evalúa en el intervalo [0, T], donde T es el período orbital del satélite.

    La función g(t) representa la diferencia entre la distancia radial r(t) y el valor r0 proporcionado.

    La función utiliza una tolerancia 'tol' para determinar si la raíz se ha encontrado con suficiente precisión.

    La función comienza definiendo la tolerancia y luego define la función g(t) que representa la diferencia entre r(t) y r0.

    Luego, se calcula el período orbital T y se establece un paso inicial para encontrar un cambio de signo en g(t).
    Se busca un intervalo [t_a, t_b] donde g(t_a) y g(t_b) tengan signos opuestos, lo que indica la presencia de una raíz en ese intervalo.
    Si se encuentra un intervalo, se aplica el método de bisección para encontrar la raíz con la tolerancia especificada.
    Si se encuentra la raíz, se retorna el instante t0 en formato astropy Time.
    Si no se encuentra un intervalo con cambio de signo o si no se encuentra la raíz, se imprime un mensaje y se retorna None.
    Si se alcanza el número máximo de iteraciones sin convergencia, se imprime un mensaje y se retorna None.
    Si se encuentra la raíz, se retorna el instante t0 en formato astropy Time.
    """
    # Definir la tolerancia
    tol = 1e-6  # km

    # Definimos g(t) = r(t) - r0
    # Usaremos una búsqueda de raíces por bisección en el intervalo [0, T].

    def g(t):
        # r(t) - r0
        t_ = tp + TimeDelta(t, format='sec')
        return position(t_)[0] - r0

    
    # Período orbital
    T = 2.0 * np.pi * np.sqrt(a**3 / GM)
    # Paso inicial para encontrar un cambio de signo
    step = 60.0  # 60 segundos

    # Hallar un intervalo [t_a, t_b] con g(t_a)*g(t_b) < 0
    t_a = 0.0
    f_a = g(t_a)
    hallo_intervalo = False

    while t_a < T:
        t_b = t_a + step
        if t_b > T:
            t_b = T
        f_b = g(t_b)

        # Revisamos si hay cambio de signo
        if f_a * f_b < 0:
            hallo_intervalo = True
            break

        t_a = t_b
        f_a = f_b

    if not hallo_intervalo:
        print("No se encontró cambio de signo para r(t) - r0 en [tp, tp+T].")
        return None
    
    # Aplicar bisección en el intervalo [t_a, t_b]

    for _ in range(200):  # Máximo 200 iteraciones
        t_m = 0.5 * (t_a + t_b)
        f_m = g(t_m)

        if abs(f_m) < tol:
            # Si la distancia es menor que la tolerancia, convergemos
            t_sol = tp + TimeDelta(t_m, format='sec')
            return t_sol

        # Ajuste del intervalo
        if f_a * f_m < 0:
            # la raíz está en [t_a, t_m]
            t_b = t_m
            f_b = f_m
        else:
            # la raíz está en [t_m, t_b]
            t_a = t_m
            f_a = f_m

    print("No hubo convergencia en la bisección para r0 =", r0)
    return None

# ============================================================================

# Test

# 1) Verificación: posición en 2025-04-01 00:00:00 UTC
t_test = Time("2025-04-01 00:00:00", format="iso", scale="utc")
r_test, phi_test = position(t_test)

print("Posición del satélite en t =", t_test.iso)
print("r(t)   = {:.12f} km".format(r_test))
print("phi(t) = {:.12f} rad  ({:.6f}°)".format(phi_test, np.degrees(phi_test)))

# 2) Graficar la órbita
#orbit()

# 3) Buscar instante en que r0 = 1.5 * RE
r0_test = 1.5 * RT
t_r0 = date(r0_test)
if t_r0 is not None:
    r_calc = position(t_r0)[0]
    print("\nr0 = {:.6f} km se alcanza en t = {}".format(r0_test, t_r0.iso))
    print("r(t_r0) = {:.9f} km (error = {:.9f} km)".format(r_calc, abs(r_calc - r0_test)))
