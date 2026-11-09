import math
import random
import numpy as np


class Auto3DEnv:
    """
    Entorno de simulación para el Auto con sensores láser (Raycasting).
    Diseñado para que el ciclo de Reinforcement Learning (reset / step)
    sea idéntico al estándar de la industria (Gymnasium).
    """

    def __init__(self):
        # Parámetros de la pista elíptica con curvas (en unidades 3D)
        self.track_rxOuter = 42.0
        self.track_rzOuter = 28.0
        self.track_rxInner = 26.0
        self.track_rzInner = 12.0

        # Ángulos de los 5 sensores láser respecto a la dirección del auto (en radianes)
        # [-60°, -30°, 0° (frente), +30°, +60°]
        self.sensor_angles = [
            -math.pi / 3,
            -math.pi / 6,
            0.0,
            math.pi / 6,
            math.pi / 3,
        ]
        self.max_sensor_dist = 18.0

        # Obstáculos estáticos sobre la pista (x, z, radio)
        # Ubicados estratégicamente dejando libre la largada en (0, -20)
        self.obstacles = [
            (24.0, -14.0, 1.8),
            (33.0, 4.0, 2.0),
            (12.0, 19.5, 1.8),
            (-15.0, 18.5, 1.8),
            (-34.0, -2.0, 2.0),
            (-22.0, -15.0, 1.8),
        ]

        self.reset()

    def reset(self):
        """Reinicia el auto en la línea de largada y devuelve el estado inicial (sensores)."""
        # Posición inicial en la parte inferior de la pista mirando hacia la derecha (+X)
        self.x = 0.0
        self.z = 16.0  # Entre radio interno (12) y externo (28), pero ojo: obstáculo en (0, 20)
        # Mejor arrancamos en ángulo -90° de la pista libre de obstáculos:
        start_angle = -math.pi / 2
        mid_rx = (self.track_rxOuter + self.track_rxInner) / 2.0
        mid_rz = (self.track_rzOuter + self.track_rzInner) / 2.0
        self.x = mid_rx * math.cos(start_angle)
        self.z = mid_rz * math.sin(start_angle)

        # Dirección tangente a la pista en sentido antihorario
        self.heading = 0.0  # Mirando hacia +X
        self.speed = 0.45
        self.steps = 0
        self.max_steps = 800
        self.done = False
        self.total_reward = 0.0
        self.prev_progress_angle = math.atan2(self.z / mid_rz, self.x / mid_rx)
        self.laps = 0

        self.sensor_readings, self.sensor_hits = self._compute_sensors()
        return self._get_state()

    def _is_point_off_track(self, px, pz):
        """Verifica si un punto (px, pz) salió de los bordes de la pista o chocó un obstáculo."""
        # Ecuación de la elipse exterior: (x/rxOuter)^2 + (z/rzOuter)^2 > 1
        norm_outer = (px / self.track_rxOuter) ** 2 + (pz / self.track_rzOuter) ** 2
        if norm_outer > 1.0:
            return True

        # Ecuación de la elipse interior: (x/rxInner)^2 + (z/rzInner)^2 < 1
        norm_inner = (px / self.track_rxInner) ** 2 + (pz / self.track_rzInner) ** 2
        if norm_inner < 1.0:
            return True

        # Obstáculos cilíndricos
        for ox, oz, orad in self.obstacles:
            if (px - ox) ** 2 + (pz - oz) ** 2 <= orad ** 2:
                return True

        return False

    def _compute_sensors(self):
        """
        Lanza 5 rayos láser desde el auto y mide la distancia hasta el borde u obstáculo más cercano.
        Devuelve:
          - readings: lista de 5 valores normalizados entre 0.0 (choque inminente) y 1.0 (despejado)
          - hits: lista de coordenadas (hx, hz) para dibujar los láseres en el visor 3D
        """
        readings = []
        hits = []
        step_size = 0.35

        for rel_angle in self.sensor_angles:
            ray_angle = self.heading + rel_angle
            cos_a = math.cos(ray_angle)
            sin_a = math.sin(ray_angle)

            dist = 0.0
            hit_x = self.x + cos_a * self.max_sensor_dist
            hit_z = self.z + sin_a * self.max_sensor_dist

            while dist < self.max_sensor_dist:
                dist += step_size
                px = self.x + cos_a * dist
                pz = self.z + sin_a * dist
                if self._is_point_off_track(px, pz):
                    hit_x = px
                    hit_z = pz
                    break

            norm_dist = min(1.0, max(0.0, dist / self.max_sensor_dist))
            readings.append(norm_dist)
            hits.append((round(hit_x, 2), round(hit_z, 2)))

        return readings, hits

    def _get_state(self):
        """
        Devuelve el vector de entrada X para la Red Neuronal:
        5 distancias de los sensores láser normalizadas en [0, 1].
        """
        return np.array(self.sensor_readings, dtype=np.float32)

    def step(self, action):
        """
        Ejecuta un paso de simulación en el entorno.
        Acciones posibles:
          0 -> Girar a la Izquierda
          1 -> Seguir Recto
          2 -> Girar a la Derecha

        Devuelve: (nuevo_estado, recompensa, terminado, info)
        """
        if self.done:
            return self._get_state(), 0.0, True, {}

        # 1. Aplicar giro según la acción elegida por la Red Neuronal
        turn_rate = 0.075
        if action == 0:
            self.heading -= turn_rate  # Izquierda
        elif action == 2:
            self.heading += turn_rate  # Derecha

        # 2. Mover el auto hacia adelante
        self.x += math.cos(self.heading) * self.speed
        self.z += math.sin(self.heading) * self.speed
        self.steps += 1

        # 3. Actualizar sensores
        self.sensor_readings, self.sensor_hits = self._compute_sensors()

        # 4. Verificar colisión
        if self._is_point_off_track(self.x, self.z):
            self.done = True
            reward = -10.0  # Castigo por chocar
            self.total_reward += reward
            return self._get_state(), reward, True, {"reason": "collision"}

        # 5. Calcular recompensa por avanzar en el sentido correcto de la pista
        mid_rx = (self.track_rxOuter + self.track_rxInner) / 2.0
        mid_rz = (self.track_rzOuter + self.track_rzInner) / 2.0
        current_angle = math.atan2(self.z / mid_rz, self.x / mid_rx)

        # Diferencia angular avanzando desde -pi/2 hacia 0 (sentido horario en el plano X-Z)
        delta_angle = current_angle - self.prev_progress_angle
        # Normalizar a [-pi, pi]
        while delta_angle > math.pi:
            delta_angle -= 2 * math.pi
        while delta_angle < -math.pi:
            delta_angle += 2 * math.pi

        self.prev_progress_angle = current_angle

        # Recompensa base por avanzar + mantener distancia segura de las paredes
        left_space = (self.sensor_readings[0] + self.sensor_readings[1]) * 0.5
        center_clearance = self.sensor_readings[2]
        right_space = (self.sensor_readings[3] + self.sensor_readings[4]) * 0.5
        min_clearance = min(self.sensor_readings)

        # Premiamos avanzar por la pista y mantener el centro despejado
        reward = 0.25 + (delta_angle * 18.0) + (0.4 * center_clearance) + (0.3 * min_clearance)

        # Si eligió girar hacia el lado más despejado cuando hay pared cerca, damos un refuerzo inmediato
        if min_clearance < 0.65:
            if action == 0 and left_space > right_space:
                reward += 0.35
            elif action == 2 and right_space > left_space:
                reward += 0.35
            elif (action == 0 and left_space < right_space) or (action == 2 and right_space < left_space):
                reward -= 0.35

        if min_clearance < 0.22:
            reward -= 0.3  # Penalización por pasar raspando la pared

        self.total_reward += reward

        if self.steps >= self.max_steps:
            self.done = True
            reward += 5.0  # Bono por sobrevivir todo el episodio

        return self._get_state(), float(reward), self.done, {}
