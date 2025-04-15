import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
from astropy import units as u
import math

# =============================================================================
#                           PARÁMETROS
# =============================================================================
# Definición de constantes y parámetros orbitales..
RT = 6378.1363 * u.km              # Radio de la Tierra
GM = 398600.4405 * (u.km**3 / u.s**2)  # Constante gravitacional de la Tierra
a = 1.30262 * RT                   # Semieje mayor de la órbita
e = 0.16561                        # Excentricidad 
omega_deg = 15.0                   # Argumento del pericentro en grados
omega = np.deg2rad(omega_deg)      # Argumento del pericentro en radianes

# Tiempo de paso por el pericentro (en formato UTC)
tp = Time("2025-03-31 00:00:00", format="iso", scale="utc")

# =============================================================================
#                           FUNCIÓN kepler
# =============================================================================
def kepler(l, e, tol=1e-14, max_iter=200):
    """
    Resuelve la ecuación de Kepler (E - e*sin(E) = l) mediante el método de Newton-Raphson.
    
    Parámetros:
      l        : anomalía media (adimensional)
      e        : excentricidad
      tol      : tolerancia para la convergencia
      max_iter : número máximo de iteraciones
      
    Retorna:
      E        : anomalía excéntrica (en radianes)
      
    Se utiliza la derivada f'(E) = 1 - e*cos(E) para la iteración:
      E(n+1)) = En - f(En)/f'(En)
    """
    # Valor inicial: se toma E = l
    E = l
    for i in range(max_iter):
        f = E - e * np.sin(E) - l
        fp = 1 - e * np.cos(E)
        if abs(fp) < 1e-15:
            break
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E

# =============================================================================
#                           FUNCIÓN position
# =============================================================================
def position(t):
    """
    Calcula la posición del satélite en un instante dado.
    
    Parámetros:
      t : instante (objeto astropy.time.Time)
      
    Retorna:
      r   : distancia radial (con unidades, por defecto en km)
      phi : ángulo (en radianes, entre 0 y 2π)
    
    Procedimiento:
      1. Calcula la anomalía media l = n * dt, donde dt es el tiempo transcurrido desde tp y
         n = sqrt(GM/a^3) es el movimiento medio.
      2. Resuelve la ecuación de Kepler para obtener la anomalía excéntrica E.
      3. A partir de E se calcula la anomalía verdadera f usando la fórmula:
             tan(f/2) = sqrt((1+e)/(1-e)) * tan(E/2)
         Se emplea np.arctan2 para asegurar el cuadrante correcto.
      4. La distancia radial se obtiene como:
             r = a*(1 - e^2) / (1 + e*cos(f))
      5. El ángulo en el plano orbital es:
             phi = (f + omega) mod 2*pi
    """
    # Convertir el tiempo transcurrido a segundos
    dt = (t - tp).to(u.s).value  
    # Calcular el movimiento medio n (en 1/s)
    n = np.sqrt((GM / a**3).to(u.s**-2).value)
    l = n * dt  # Anomalía media 

    # Resolver la ecuación de Kepler para obtener la anomalía excéntrica E
    E = kepler(l, e)

    # Calcular la anomalía verdadera f usando la relación:
    # tan(f/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    sin_E2 = np.sin(E / 2)
    cos_E2 = np.cos(E / 2)
    num = np.sqrt(1 + e) * sin_E2
    den = np.sqrt(1 - e) * cos_E2
    f = 2.0 * np.arctan2(num, den)
    f %= 2 * np.pi  # Asegurar que f esté en [0, 2π)

    # Calcular la distancia radial r
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    
    # Calcular el ángulo phi
    phi = (f + omega) % (2 * np.pi)

    return r, phi

# =============================================================================
#                           FUNCIÓN orbit
# =============================================================================
def orbit():
    """
    Grafica la órbita completa del satélite a lo largo de un período.
    
    Se calcula el período orbital T usando la ley de Kepler:
      T = 2π * sqrt(a^3/GM)
    Luego se generan n puntos igualmente espaciados en tiempo para obtener la trayectoria.
    La posición se transforma de coordenadas polares (r, phi) a cartesianas (x, y) para la gráfica.
    """
    n_points = 1000  # Número de puntos para la trayectoria
    
    # Calcular el período orbital T (en segundos)
    T_val = 2 * np.pi * np.sqrt((a**3 / GM).to(u.s**2).value)
    dt = np.linspace(0, T_val, n_points)  # Intervalo de tiempo en segundos
    tiempos = [tp + TimeDelta(d, format='sec') for d in dt]
    
    r_vals, phi_vals = [], []
    for t in tiempos:
        r_val, phi_val = position(t)
        r_vals.append(r_val.value)  # Extraer el valor numérico (km)
        phi_vals.append(phi_val)
    
    r_vals = np.array(r_vals)
    phi_vals = np.array(phi_vals)
    
    # Conversión a coordenadas cartesianas
    x_vals = r_vals * np.cos(phi_vals)
    y_vals = r_vals * np.sin(phi_vals)
    
    # Configuración y visualización del gráfico
    plt.figure(figsize=(8, 8))
    plt.plot(x_vals, y_vals, label='Órbita del satélite')
    plt.plot(0, 0, 'ro', label='Tierra')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title('Órbita del satélite')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()

# =============================================================================
#                           FUNCIÓN date
# =============================================================================
def date(r0):
    """
    Dado un valor de distancia radial r0, encuentra el instante t0 (entre tp y tp+T)
    en el cual el satélite se encuentra a esa distancia.
    
    Se define la función:
        g(t) = r(t) - r0
    y se aplica el método de bisección para hallar t tal que g(t) = 0 con una tolerancia predefinida.
    
    Parámetros:
      r0 : distancia radial (con unidades, por ejemplo, en km)
    
    Retorna:
      t0 : instante en el que r(t0) = r0 (objeto astropy.time.Time) o None si no se halla
    """
    # Tolerancia en km
    tol = 1e-6
    
    # Validación previa: comprobar si r0 está dentro del rango orbital posible
    r_min = a * (1 - e)
    r_max = a * (1 + e)
    if not (r_min <= r0 <= r_max):
        print(f"ERROR: r0 = {r0} está fuera del rango permitido [{r_min}, {r_max}] km.")
        return None
    
    # Función g(t) que devuelve la diferencia entre r(t) y r0, donde t se expresa en segundos
    def g(t_sec):
        t_current = tp + TimeDelta(t_sec, format='sec')
        return position(t_current)[0].value - r0.value  # convertir a valor numérico
    
    # Calcular el período orbital T (en segundos)
    T_val = 2 * np.pi * np.sqrt((a**3 / GM).to(u.s**2).value)
    
    # Buscar un intervalo [t_a, t_b] en el que g(t) cambie de signo.
    step = 60.0  # incremento de 60 segundos
    t_a = 0.0
    f_a = g(t_a)
    intervalo_encontrado = False

    while t_a < T_val:
        t_b = t_a + step
        if t_b > T_val:
            t_b = T_val
        f_b = g(t_b)
        if f_a * f_b < 0:
            intervalo_encontrado = True
            break
        t_a = t_b
        f_a = f_b

    if not intervalo_encontrado:
        print("No se encontró un intervalo con cambio de signo para r(t) - r0 en [tp, tp+T].")
        return None

    # Aplicar el método de bisección en el intervalo [t_a, t_b]
    for _ in range(200):  # máximo 200 iteraciones
        t_m = 0.5 * (t_a + t_b)
        f_m = g(t_m)
        if abs(f_m) < tol:
            t_sol = tp + TimeDelta(t_m, format='sec')
            return t_sol
        if f_a * f_m < 0:
            t_b = t_m
            # f_b se actualiza a f_m
        else:
            t_a = t_m
            f_a = f_m

    print("La bisección no convergió para r0 =", r0)
    return None

# =============================================================================
#                                PRUEBAS
# =============================================================================
# Verificar la posición en un instante dado
t_test = Time("2025-04-01 00:00:00", format="iso", scale="utc")
r_test, phi_test = position(t_test)
print("Posición en t =", t_test.iso)
print(f"r(t)   = {r_test.value} km")
print(f"phi(t) = {phi_test} rad ({np.degrees(phi_test)}°)")

# Graficar la órbita (descomentar la siguiente línea para visualizar la gráfica)
# orbit()

# Buscar el instante en que se alcanza r0 = 1.5 * RT
r0_test = 1.5 * RT
t_r0 = date(r0_test)
if t_r0 is not None:
    r_calc = position(t_r0)[0]
    print(f"r0 = {r0_test.value} km se alcanza en t = {t_r0.iso}")
    print(f"r(t_r0) = {r_calc.value} km (error = {abs(r_calc.value - r0_test.value)} km)")
