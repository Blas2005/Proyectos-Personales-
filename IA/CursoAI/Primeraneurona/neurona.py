import numpy as np

# ==========================================
# PARTE 1: Función de Activación
# ==========================================

    
    ### TU CÓDIGO AQUÍ ###

def sigmoide (z):
    s = 1/(1 + np.exp(-z))
    return s


# ==========================================
# PARTE 2: Inicialización de Parámetros
# ==========================================
def inicializar_parametros(dimension):
    """
    Crea un vector w de ceros de dimensión (dimension, 1) y b inicializado en 0.0.
    """
    ### TU CÓDIGO AQUÍ ###

    w = np.zeros((dimension,1))
    b = 0.0
    
    ### FIN ###
    return w, b


# ==========================================
# PARTE 3: Propagación (Forward y Backward)
# ==========================================
def propagacion(w, b, X, Y):
    """
    Implementa la función de coste y sus gradientes.
    Argumentos:
        w: Pesos, matriz de tamaño (n, 1)
        b: Sesgo (escalar)
        X: Datos de entrada de tamaño (n, m)
        Y: Etiquetas reales (0 o 1) de tamaño (1, m)
    """
    m = X.shape[1]

    # --- FORWARD PASS (De entradas a Coste) ---
    # 1. z = w^T * X + b
    # 2. a = sigmoide(z)
    # 3. coste = (-1/m) * sum( Y * ln(a) + (1 - Y) * ln(1 - a) )
    ### TU CÓDIGO AQUÍ ###

    z = np.dot(w.T, X) + b
    a = sigmoide(z)
    coste = (-1/m) * np.sum(Y * np.log(a) + (1 - Y) * np.log(1-a))
    ### FIN ###

    # --- BACKWARD PASS (Gradientes) ---
    # 4. dz = a - Y
    # 5. dw = (1/m) * X * dz^T
    # 6. db = (1/m) * sum(dz)
    ### TU CÓDIGO AQUÍ ###

    dz = a - Y
    dw = (1/m) * np.dot(X, dz.T)
    db = (1/m) * np.sum(dz)

    
    ### FIN ###

    gradientes = {"dw": dw, "db": db}
    return gradientes, coste


# ==========================================
# PARTE 4: Bucle de Entrenamiento (Optimización)
# ==========================================
def optimizar(w, b, X, Y, num_iteraciones, tasa_aprendizaje):
    """
    Actualiza w y b ejecutando el descenso por gradiente.
    """
    costes = []

    for i in range(num_iteraciones):
        # 1. Calcular gradientes y coste actual llamando a propagacion()
        gradientes, coste = propagacion(w, b, X, Y)
        
        dw = gradientes["dw"]
        db = gradientes["db"]

        # 2. Actualizar parámetros usando la regla: param = param - tasa * dparam
        ### TU CÓDIGO AQUÍ ###
        w = w - tasa_aprendizaje * dw
        b = b - tasa_aprendizaje * db
        ### FIN ###

        # Registro de coste cada 100 pasos
        if i % 100 == 0:
            costes.append(coste)
            print(f"Coste en iteración {i}: {coste}")

    parametros = {"w": w, "b": b}
    return parametros, costes


# ==========================================
# PARTE 5: Test de Verificación
# ==========================================
if __name__ == "__main__":
    print("--- Probando implementación con datos sintéticos ---")
    
    # Matriz X de 2 características y 3 ejemplos (dimensión 2x3)
    X_test = np.array([[1.0, 2.0, -1.0], 
                       [3.0, 4.0, -3.2]])
    # Etiquetas reales para los 3 ejemplos (dimensión 1x3)
    Y_test = np.array([[1, 1, 0]])

    # Inicializar y entrenar por 200 iteraciones
    w_ini, b_ini = inicializar_parametros(dimension=2)
    params, historial_coste = optimizar(w_ini, b_ini, X_test, Y_test, num_iteraciones=2000, tasa_aprendizaje=0.01)

    print("\nEntrenamiento finalizado.")
    print("w resultante:\n", params["w"])
    print("b resultante:\n", params["b"])