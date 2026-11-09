import numpy as np


class RedNeuronalRL:
    """
    ===========================================================================
    DE REGRESIÓN LOGÍSTICA A REINFORCEMENT LEARNING (MULTICAPA EN NUMPY)
    ===========================================================================
    En Regresión Logística tenías una sola capa:
        z = X @ W + b
        y_hat = sigmoide(z)
        dz = y_hat - y_real

    Aquí extendemos esa misma idea a 2 capas para controlar el Auto 3D:
      - Entradas (X): 5 sensores láser (distancias entre 0.0 y 1.0)
      - Capa Oculta 1: 16 neuronas con activación tanh(z1)
      - Capa de Salida: 3 neuronas con Softmax(z2) -> Probabilidad de:
            [0: Girar Izquierda, 1: Seguir Recto, 2: Girar Derecha]
    """

    def __init__(self, n_inputs=5, n_hidden=16, n_actions=3, learning_rate=0.015, gamma=0.96):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma  # Factor de descuento para recompensas futuras

        # 1. Inicialización de Pesos (W) y Sesgos (b)
        # Usamos inicialización Xavier/Glorot para que las señales no exploten ni se apaguen
        self.W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(1.0 / n_inputs)
        self.b1 = np.zeros((1, n_hidden))

        self.W2 = np.random.randn(n_hidden, n_actions) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros((1, n_actions))

        # Baseline móvil de retorno para saber si un episodio fue mejor o peor que lo habitual
        self.baseline_retorno = 0.0

        # Memoria del episodio actual (para entrenar al terminar cada intento)
        self.memoria_estados = []
        self.memoria_ocultas = []
        self.memoria_probs = []
        self.memoria_acciones = []
        self.memoria_recompensas = []

    @staticmethod
    def softmax(z):
        """
        Generalización de la función Sigmoide para más de 2 opciones.
        Convierte puntajes crudos (logits) en probabilidades que suman 1.0 (100%).
        """
        z_estable = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_estable)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def forward(self, estado_x):
        """
        PROPAGACIÓN HACIA ADELANTE (Forward Pass):
        Dado el vector de 5 sensores, calcula las activaciones ocultas y
        las probabilidades de las 3 acciones.
        """
        # Centramos las lecturas de [0, 1] a [-1, 1] para acelerar el aprendizaje
        x = (estado_x.reshape(1, -1) - 0.5) * 2.0  # Forma (1, 5)

        # Paso 1: Capa Oculta (igual que regresión lineal + activación no lineal tanh)
        z1 = np.dot(x, self.W1) + self.b1      # Forma (1, 16)
        a1 = np.tanh(z1)                       # Valores entre -1 y 1

        # Paso 2: Capa de Salida (logits -> probabilidades con Softmax)
        z2 = np.dot(a1, self.W2) + self.b2     # Forma (1, 3)
        probs = self.softmax(z2)               # Forma (1, 3), suman 1.0

        return a1[0], probs[0]

    def elegir_accion(self, estado_x):
        """
        Pasa los sensores por la red y elige una acción muestreando según sus probabilidades.
        Así el auto explora al principio y explota lo aprendido a medida que mejora.
        """
        a1, probs = self.forward(estado_x)
        accion = np.random.choice(self.n_actions, p=probs)

        # Guardamos en memoria lo que ocurrió en este frame para el Backpropagation
        self.memoria_estados.append((estado_x - 0.5) * 2.0)
        self.memoria_ocultas.append(a1)
        self.memoria_probs.append(probs)
        self.memoria_acciones.append(accion)

        return int(accion), a1, probs

    def registrar_recompensa(self, recompensa):
        """Guarda la recompensa recibida tras ejecutar la acción."""
        self.memoria_recompensas.append(recompensa)

    def calcular_recompensas_descontadas(self):
        """
        En Reinforcement Learning, una buena decisión hoy vale por las recompensas
        que permite alcanzar en los pasos siguientes:
            G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} ...
        """
        G = np.zeros(len(self.memoria_recompensas), dtype=np.float32)
        acumulado = 0.0
        for t in reversed(range(len(self.memoria_recompensas))):
            acumulado = self.memoria_recompensas[t] + 0.82 * acumulado
            G[t] = acumulado

        # Normalizamos las ventajas del episodio para que las mejores decisiones
        # tengan G > 0 y los errores previos al choque tengan G < 0
        if len(G) > 1:
            G = (G - np.mean(G)) / (np.std(G) + 1e-6)
        return G

    def entrenar_episodio(self):
        """
        =======================================================================
        BACKPROPAGATION POR REFUERZO (Algoritmo REINFORCE / Policy Gradient)
        =======================================================================
        ¡Notá la similitud exacta con Regresión Logística!:
        En Regresión Logística:
            dz2 = (y_hat - y_etiqueta)
        En Reinforcement Learning no tenemos "etiqueta correcta", sino que usamos
        la acción que tomamos multiplicada por qué tan buena fue su recompensa (G):
            dz2 = (probs - one_hot(accion)) * G
        """
        if not self.memoria_estados:
            return 0.0

        X = np.vstack(self.memoria_estados)          # Forma (N, 5)
        A1 = np.vstack(self.memoria_ocultas)         # Forma (N, 16)
        Probs = np.vstack(self.memoria_probs)        # Forma (N, 3)
        Acciones = np.array(self.memoria_acciones)   # Forma (N,)
        G = self.calcular_recompensas_descontadas().reshape(-1, 1)  # Forma (N, 1)

        N = len(X)

        # Convertimos la acción tomada en vector one-hot: ej. acción 2 -> [0, 0, 1]
        Y_tomada = np.zeros_like(Probs)
        Y_tomada[np.arange(N), Acciones] = 1.0

        # --- PASO 1: Gradiente en la Capa de Salida (Capa 2) ---
        # Si G > 0, empuja Probs hacia Y_tomada. Si G < 0, la aleja.
        dZ2 = (Probs - Y_tomada) * G                 # Forma (N, 3)
        dW2 = np.dot(A1.T, dZ2) / N                  # Forma (16, 3)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / N # Forma (1, 3)

        # --- PASO 2: Propagar el error hacia atrás a la Capa Oculta (Capa 1) ---
        # La derivada de tanh(z1) es (1 - A1^2)
        dA1 = np.dot(dZ2, self.W2.T)                 # Forma (N, 16)
        dZ1 = dA1 * (1.0 - A1 ** 2)                  # Forma (N, 16)
        dW1 = np.dot(X.T, dZ1) / N                   # Forma (5, 16)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / N # Forma (1, 16)

        # --- PASO 3: Descenso del Gradiente (Actualizar Pesos) ---
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Limpiamos la memoria para el próximo episodio
        self.memoria_estados.clear()
        self.memoria_ocultas.clear()
        self.memoria_probs.clear()
        self.memoria_acciones.clear()
        self.memoria_recompensas.clear()
