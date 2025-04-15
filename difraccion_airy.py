import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def simpson(x, f):
    """
    Realiza la integración numérica utilizando la regla de Simpson compuesta.
    
    Parámetros:
        x: array de puntos en el eje de integración.
        f: array de valores de la función a integrar.
        
    Retorna:
        Valor aproximado de la integral.
    """
    n = len(x)
    integral = 0.0
    i = 0
    while 2*i + 2 < n:
        dx = x[2*i + 1] - x[2*i]
        integral += dx * (f[2*i] + 4*f[2*i + 1] + f[2*i + 2]) / 3
        i += 1
    # Si el número de subintervalos no es par, se aplica la regla del trapecio en el último intervalo
    if 2*i + 2 != n - 1:
        dx = x[-1] - x[-2]
        f_mean = (f[-1] + f[-2]) / 2
        integral += dx * f_mean
    return integral

def bessel_J1_simpson(x, N=1000):
    """
    Calcula la función de Bessel de primer tipo y orden 1 mediante integración numérica.
    
    Parámetros:
        x: argumento de la función de Bessel.
        N: número de puntos para la integración (por defecto 1000).
        
    Retorna:
        Aproximación de J_1(x) utilizando la regla de Simpson.
    """
    if N % 2 == 1:
        N += 1  # Se asegura que N sea par
    theta = np.linspace(0, np.pi, N + 1)
    integrand = np.cos(theta - x * np.sin(theta))
    return simpson(theta, integrand) / np.pi

def bessel_Jm_simpson(m, x, N=1000):
    """
    Calcula la función de Bessel de primer tipo y orden m mediante integración numérica.
    
    Parámetros:
        m: orden de la función de Bessel (entero no negativo).
        x: argumento de la función.
        N: número de puntos para la integración (por defecto 1000).
        
    Retorna:
        Aproximación de J_m(x) utilizando la regla de Simpson.
    """
    if N % 2 == 1:
        N += 1
    theta = np.linspace(0, np.pi, N + 1)
    integrand = np.cos(m * theta - x * np.sin(theta))
    return simpson(theta, integrand) / np.pi

def plot_bessel_functions():
    """
    Grafica las funciones de Bessel de primer tipo para los órdenes m = 0, 1 y 2 en el intervalo x ∈ [0, 20].
    """
    # Valores de x para graficar
    x_values = np.linspace(0, 20, 200)
    # Órdenes de Bessel a graficar
    orders = [0, 1, 2]

    plt.figure(figsize=(8, 6))
    for m in orders:
        # Se calcula J_m(x) para cada valor de x utilizando la función definida
        Jm_values = [bessel_Jm_simpson(m, x, N=1000) for x in x_values]
        plt.plot(x_values, Jm_values, label=f'J_{m}(x)')
    plt.axhline(0, color='black', linewidth=0.8, linestyle="--")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$J_m(x)$')
    plt.title('Funciones de Bessel de primer tipo')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_diffraction_pattern():
    """
    Genera un gráfico de densidad para la intensidad del patrón de difracción circular 
    para una fuente puntual con λ = 500 nm en la región 0 ≤ r ≤ 2 μm.
    
    La intensidad se define como:
        I(r) = [2 J_1(k r) / (k r)]^2,
    donde k = 2π/λ y se adapta para trabajar con r en micrómetros.
    """
    # Parámetros físicos
    lambda_nm = 500                 # Longitud de onda en nanómetros
    lambda_m = lambda_nm * 1e-9       # Conversión a metros
    k = 2 * np.pi / lambda_m          # Número de onda en m^-1
    k_eff = k * 1e-6                  # k_eff en (μm)^-1, para trabajar con r en μm

    # Configuración de la malla en el plano focal
    num_points = 600                # Resolución de la malla
    lim = 1.0                       # Límites en μm para x e y
    x = np.linspace(-lim, lim, num_points)
    y = np.linspace(-lim, lim, num_points)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)         # Distancia radial en μm

    # Cálculo de la intensidad I(r) en cada punto de la malla
    I = np.zeros_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            r_val = R[i, j]
            if r_val == 0:
                I[i, j] = 1.0  # Se define I(0) = 1 para evitar división por cero
            else:
                J1_val = bessel_J1_simpson(k_eff * r_val, N=5000)
                I[i, j] = (2 * J1_val / (k_eff * r_val))**2

    # Graficar el patrón de difracción con escala logarítmica para resaltar detalles
    plt.figure(figsize=(7, 7))
    plt.imshow(I, extent=[-lim, lim, -lim, lim], origin='lower',
               cmap='inferno',
               norm=colors.LogNorm(vmin=1e-5, vmax=I.max()),
               interpolation='bilinear')
    plt.colorbar(label='Intensidad (escala log)')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.title('Patrón de difracción (Airy Pattern)\nλ = 500 nm')
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()

def find_first_four_maxima():
    """
    Localiza los 4 primeros máximos del patrón de intensidad I(r) en el intervalo 0 ≤ r ≤ 2 μm.
    
    La intensidad se calcula como:
        I(r) = [2 J_1(k r) / (k r)]^2.
        
    Retorna:
        - Una lista con los pares (r, I(r)) correspondientes a los 4 primeros máximos.
        - Los vectores r_vals e I_vals para fines de graficación.
    """
    # Parámetros físicos
    lambda_nm = 500                 # nm
    lambda_m = lambda_nm * 1e-9       # m
    k = 2 * np.pi / lambda_m          # m^-1
    k_eff = k * 1e-6                  # (μm)^-1

    # Definición del intervalo para r y resolución
    r_vals = np.linspace(0, 2, 10000)
    I_vals = np.zeros_like(r_vals)

    # Cálculo de I(r) para cada valor de r
    for i, r in enumerate(r_vals):
        if r == 0:
            I_vals[i] = 1.0
        else:
            J1_val = bessel_J1_simpson(k_eff * r, N=5000)
            I_vals[i] = (2 * J1_val / (k_eff * r))**2

    # Identificación de máximos locales comparando con los vecinos
    maxima_indices = []
    for i in range(1, len(I_vals) - 1):
        if I_vals[i] > I_vals[i - 1] and I_vals[i] > I_vals[i + 1]:
            maxima_indices.append(i)

    # Seleccionar los 4 primeros máximos
    first_four = maxima_indices[:4]
    maxima_points = [(r_vals[i], I_vals[i]) for i in first_four]

    return maxima_points, r_vals, I_vals, first_four

def plot_maxima():
    """
    Grafica la intensidad I(r) y marca los 4 primeros máximos encontrados en el intervalo 0 ≤ r ≤ 2 μm.
    También muestra por consola los valores de r y I(r) correspondientes a cada máximo.
    """
    maxima_points, r_vals, I_vals, first_four_indices = find_first_four_maxima()
    
    print("Los valores de r para los 4 primeros máximos son:")
    for idx, (r_val, intensity) in enumerate(maxima_points, start=1):
        print(f"Máximo {idx}: r = {r_val:.6f} μm, I(r) = {intensity:.6f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(r_vals, I_vals, label="I(r)")
    plt.plot([pt[0] for pt in maxima_points],
             [pt[1] for pt in maxima_points],
             "ro", label="Máximos")
    plt.xlabel("r (μm)")
    plt.ylabel("Intensidad I(r)")
    plt.title("Patrón de difracción: Máximos de intensidad")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    Función principal que organiza la ejecución del proyecto según los requerimientos:
        1. Gráfica de las funciones de Bessel.
        2. Gráfica del patrón de difracción circular.
        3. Localización y graficación de los 4 primeros máximos del patrón de intensidad.
    """
    # Gráfica de las funciones de Bessel para m = 0, 1 y 2
    plot_bessel_functions()
    
    # Gráfica del patrón de difracción en 2D para λ = 500 nm
    plot_diffraction_pattern()
    
    # Localización y graficación de los 4 primeros máximos de la intensidad
    plot_maxima()

if __name__ == '__main__':
    main()
