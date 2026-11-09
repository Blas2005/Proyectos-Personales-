import os
import sys
import math
import numpy as np
import pygame

from environment import GridWorldEnv
from agent import QLearningAgent

# Configuración de ventana y colores
CELL_SIZE = 48
GRID_ROWS = 12
GRID_COLS = 16
GRID_WIDTH = GRID_COLS * CELL_SIZE
GRID_HEIGHT = GRID_ROWS * CELL_SIZE
PANEL_WIDTH = 360
WINDOW_WIDTH = GRID_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = max(GRID_HEIGHT, 640)

# Paleta de colores estilo moderno oscuro
BG_COLOR = (18, 22, 31)
GRID_BG = (26, 32, 44)
GRID_LINE = (42, 50, 68)
WALL_COLOR = (74, 85, 104)
WALL_HIGHLIGHT = (100, 116, 139)
TRAP_COLOR = (229, 62, 62)
START_COLOR = (49, 130, 206)
GOAL_COLOR = (56, 161, 105)
AGENT_COLOR = (246, 224, 94)
PANEL_BG = (23, 28, 40)
PANEL_BORDER = (45, 55, 72)
TEXT_MAIN = (237, 242, 247)
TEXT_MUTED = (160, 174, 192)
BTN_BG = (45, 55, 72)
BTN_HOVER = (66, 80, 102)
BTN_ACTIVE = (49, 130, 206)
ACCENT_GREEN = (72, 187, 120)
ACCENT_RED = (245, 101, 101)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_data")
BRAIN_FILE = os.path.join(SAVE_DIR, "agent_brain.json")
MAP_FILE = os.path.join(SAVE_DIR, "custom_map.json")


class Button:
    def __init__(self, x, y, w, h, text, callback, toggle=False, active=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.toggle = toggle
        self.active = active
        self.hovered = False

    def draw(self, surface, font):
        if self.active:
            color = BTN_ACTIVE
        elif self.hovered:
            color = BTN_HOVER
        else:
            color = BTN_BG

        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, width=1, border_radius=6)

        txt_surf = font.render(self.text, True, TEXT_MAIN)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.toggle:
                    self.active = not self.active
                self.callback()
                return True
        return False


class LabApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Grid World Lab - Entrenamiento Q-Learning en Vivo")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_sub = pygame.font.SysFont("segoeui", 15, bold=True)
        self.font_body = pygame.font.SysFont("segoeui", 14)
        self.font_small = pygame.font.SysFont("segoeui", 12)

        self.env = GridWorldEnv(rows=GRID_ROWS, cols=GRID_COLS)
        self.agent = QLearningAgent(
            n_states=self.env.observation_space.n,
            n_actions=self.env.action_space.n,
        )

        # Estado de simulación
        self.state, _ = self.env.reset()
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.paused = False
        self.show_heatmap = True
        self.show_arrows = True

        # Velocidades: 1 paso cada N ms o N pasos por frame
        self.speed_modes = [
            ("x1 (Lento)", 1, 12),
            ("x5 (Normal)", 1, 60),
            ("x25 (Rápido)", 10, 60),
            ("x100 (Turbo)", 50, 60),
            ("x500 (Ultra)", 250, 60),
        ]
        self.speed_idx = 1

        # Herramienta de dibujo seleccionada
        self.current_tool = GridWorldEnv.WALL
        self.mouse_drawing = False
        self.erase_mode = False
        self.status_msg = "Listo. Dibujá con el mouse en la cuadrícula."
        self.status_timer = 180

        self._setup_ui()

    def _set_status(self, msg: str):
        self.status_msg = msg
        self.status_timer = 210

    def _setup_ui(self):
        px = GRID_WIDTH + 16
        pw = PANEL_WIDTH - 32
        self.buttons = []

        # Herramientas de edición (fila 1 y 2)
        bw = (pw - 8) // 2
        y = 148
        self.tool_buttons = {}

        def make_tool_cb(tool_id, name):
            def cb():
                self.current_tool = tool_id
                for tid, b in self.tool_buttons.items():
                    b.active = (tid == tool_id)
                self._set_status(f"Herramienta: {name}")
            return cb

        tools = [
            (GridWorldEnv.WALL, "Pared (1)"),
            (GridWorldEnv.TRAP, "Trampa (2)"),
            (GridWorldEnv.GOAL, "Mover Meta (3)"),
            (GridWorldEnv.START, "Mover Inicio (4)"),
        ]
        for i, (tid, label) in enumerate(tools):
            bx = px + (i % 2) * (bw + 8)
            by = y + (i // 2) * 34
            btn = Button(bx, by, bw, 28, label, make_tool_cb(tid, label), active=(tid == self.current_tool))
            self.tool_buttons[tid] = btn
            self.buttons.append(btn)

        # Controles de Simulación
        y = 248
        self.btn_pause = Button(px, y, bw, 30, "Pausar (Espacio)", self.toggle_pause, toggle=True)
        self.btn_speed = Button(px + bw + 8, y, bw, 30, f"Vel: {self.speed_modes[self.speed_idx][0]}", self.cycle_speed)
        self.buttons.extend([self.btn_pause, self.btn_speed])

        # Exploración Epsilon (+/-)
        y = 286
        third_w = (pw - 12) // 3
        self.buttons.append(Button(px, y, third_w, 28, "Explorar -", lambda: self.adjust_epsilon(-0.05)))
        self.buttons.append(Button(px + third_w + 6, y, third_w, 28, "Explorar +", lambda: self.adjust_epsilon(0.05)))
        self.btn_autodecay = Button(
            px + 2 * (third_w + 6), y, third_w, 28, "Auto-Decay", self.toggle_autodecay, toggle=True, active=True
        )
        self.buttons.append(self.btn_autodecay)

        # Visualización
        y = 322
        self.btn_heat = Button(px, y, bw, 28, "Mapa Calor Q", self.toggle_heatmap, toggle=True, active=True)
        self.btn_arrows = Button(px + bw + 8, y, bw, 28, "Flechas Política", self.toggle_arrows, toggle=True, active=True)
        self.buttons.extend([self.btn_heat, self.btn_arrows])

        # Acciones de Reinicio y Mapa
        y = 358
        self.buttons.append(Button(px, y, bw, 28, "Reiniciar IA (R)", self.reset_brain))
        self.buttons.append(Button(px + bw + 8, y, bw, 28, "Limpiar Mapa (C)", self.clear_map))

        # Guardar / Cargar
        y = 394
        self.buttons.append(Button(px, y, bw, 28, "Guardar Todo (S)", self.save_all))
        self.buttons.append(Button(px + bw + 8, y, bw, 28, "Cargar Todo (L)", self.load_all))

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.active = self.paused
        self.btn_pause.text = "Reanudar (Espacio)" if self.paused else "Pausar (Espacio)"
        self._set_status("Simulación en pausa." if self.paused else "Simulación en marcha.")

    def cycle_speed(self):
        self.speed_idx = (self.speed_idx + 1) % len(self.speed_modes)
        self.btn_speed.text = f"Vel: {self.speed_modes[self.speed_idx][0]}"
        self._set_status(f"Velocidad cambiada a {self.speed_modes[self.speed_idx][0]}")

    def adjust_epsilon(self, delta: float):
        self.agent.epsilon = float(np.clip(self.agent.epsilon + delta, 0.0, 1.0))
        self._set_status(f"Tasa de exploración (Epsilon): {self.agent.epsilon:.2f}")

    def toggle_autodecay(self):
        self.agent.auto_decay = self.btn_autodecay.active
        self._set_status(f"Auto-Decay de exploración: {'Activado' if self.agent.auto_decay else 'Desactivado'}")

    def toggle_heatmap(self):
        self.show_heatmap = self.btn_heat.active

    def toggle_arrows(self):
        self.show_arrows = self.btn_arrows.active

    def reset_brain(self):
        self.agent.reset_brain()
        self.state, _ = self.env.reset()
        self.ep_reward = 0.0
        self.ep_steps = 0
        self._set_status("Cerebro de la IA reiniciado desde cero.")

    def clear_map(self):
        self.env.clear_map()
        self.state, _ = self.env.reset()
        self.ep_reward = 0.0
        self.ep_steps = 0
        self._set_status("Mapa limpiado.")

    def save_all(self):
        self.agent.save_brain(BRAIN_FILE)
        self.env.save_map(MAP_FILE)
        self._set_status("¡Cerebro y Mapa guardados en saved_data/!")

    def load_all(self):
        b_ok = self.agent.load_brain(BRAIN_FILE)
        m_ok = self.env.load_map(MAP_FILE)
        if b_ok or m_ok:
            self.state, _ = self.env.reset()
            self.ep_reward = 0.0
            self.ep_steps = 0
            self._set_status("¡Cerebro y Mapa cargados con éxito!")
        else:
            self._set_status("No se encontraron archivos guardados aún.")

    def step_simulation(self):
        action = self.agent.select_action(self.state, training=True)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        self.agent.update(self.state, action, reward, next_state, terminated)
        self.state = next_state
        self.ep_reward += reward
        self.ep_steps += 1

        if done:
            won = (self.env.agent_pos == self.env.goal_pos)
            self.agent.end_episode(self.ep_reward, self.ep_steps, won)
            self.state, _ = self.env.reset()
            self.ep_reward = 0.0
            self.ep_steps = 0

    def handle_grid_click(self, pos, right_click=False):
        mx, my = pos
        if mx >= GRID_WIDTH or my >= GRID_HEIGHT:
            return
        c = mx // CELL_SIZE
        r = my // CELL_SIZE
        if right_click:
            self.env.set_cell(r, c, GridWorldEnv.EMPTY)
        else:
            self.env.set_cell(r, c, self.current_tool)

    def draw_grid(self):
        # Fondo del grid
        pygame.draw.rect(self.screen, GRID_BG, (0, 0, GRID_WIDTH, GRID_HEIGHT))

        # Calcular valores máximos y mínimos de Q para el mapa de calor
        max_q_per_state = np.max(self.agent.q_table, axis=1)
        q_abs_max = max(float(np.max(np.abs(max_q_per_state))), 1.0)

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cell = self.env.grid[r, c]
                state_idx = self.env.pos_to_state((r, c))

                # 1. Dibujar celda base o mapa de calor Q
                if cell == GridWorldEnv.WALL:
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
                    pygame.draw.rect(self.screen, WALL_HIGHLIGHT, rect.inflate(-6, -6), border_radius=4)
                elif cell == GridWorldEnv.TRAP:
                    pygame.draw.rect(self.screen, (70, 25, 30), rect)
                    # Icono de trampa (X / rombo rojo)
                    center = rect.center
                    pygame.draw.circle(self.screen, TRAP_COLOR, center, CELL_SIZE // 3)
                    pygame.draw.circle(self.screen, (255, 180, 180), center, CELL_SIZE // 6)
                else:
                    if self.show_heatmap:
                        q_val = float(max_q_per_state[state_idx])
                        if abs(q_val) > 0.05:
                            intensity = min(abs(q_val) / q_abs_max, 1.0)
                            if q_val > 0:
                                color = (
                                    int(26 + 30 * intensity),
                                    int(32 + 135 * intensity),
                                    int(44 + 65 * intensity),
                                )
                            else:
                                color = (
                                    int(26 + 150 * intensity),
                                    int(32 + 15 * intensity),
                                    int(44 + 20 * intensity),
                                )
                            pygame.draw.rect(self.screen, color, rect)

                # 2. Dibujar flechas de política aprendida
                if self.show_arrows and cell == GridWorldEnv.EMPTY and (r, c) != self.env.goal_pos:
                    q_vals = self.agent.q_table[state_idx]
                    if np.max(np.abs(q_vals)) > 0.01:
                        best_a = int(np.argmax(q_vals))
                        self._draw_policy_arrow(rect.center, best_a)

                # Bordes de cuadrícula
                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

        # Dibujar rastro reciente del agente
        for i, pos in enumerate(self.env.trail[:-1]):
            alpha_ratio = (i + 1) / max(len(self.env.trail), 1)
            tr_rect = pygame.Rect(
                pos[1] * CELL_SIZE + CELL_SIZE // 3,
                pos[0] * CELL_SIZE + CELL_SIZE // 3,
                CELL_SIZE // 3,
                CELL_SIZE // 3,
            )
            color = (
                int(40 + 120 * alpha_ratio),
                int(110 + 90 * alpha_ratio),
                int(160 + 50 * alpha_ratio),
            )
            pygame.draw.rect(self.screen, color, tr_rect, border_radius=4)

        # Dibujar Inicio
        sr, sc = self.env.start_pos
        s_rect = pygame.Rect(sc * CELL_SIZE + 6, sr * CELL_SIZE + 6, CELL_SIZE - 12, CELL_SIZE - 12)
        pygame.draw.rect(self.screen, START_COLOR, s_rect, border_radius=8)
        s_txt = self.font_small.render("INICIO", True, TEXT_MAIN)
        self.screen.blit(s_txt, s_txt.get_rect(center=s_rect.center))

        # Dibujar Meta
        gr, gc = self.env.goal_pos
        g_rect = pygame.Rect(gc * CELL_SIZE + 5, gr * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)
        pygame.draw.rect(self.screen, GOAL_COLOR, g_rect, border_radius=8)
        pygame.draw.rect(self.screen, (154, 230, 180), g_rect, width=2, border_radius=8)
        g_txt = self.font_small.render("META", True, (255, 255, 255))
        self.screen.blit(g_txt, g_txt.get_rect(center=g_rect.center))

        # Dibujar Agente IA (personaje con ojitos)
        ar, ac = self.env.agent_pos
        ax = ac * CELL_SIZE + CELL_SIZE // 2
        ay = ar * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 7
        pygame.draw.circle(self.screen, (180, 140, 20), (ax, ay + 2), radius)
        pygame.draw.circle(self.screen, AGENT_COLOR, (ax, ay), radius)
        pygame.draw.circle(self.screen, (30, 30, 40), (ax - 5, ay - 3), 3)
        pygame.draw.circle(self.screen, (30, 30, 40), (ax + 5, ay - 3), 3)

    def _draw_policy_arrow(self, center, action: int):
        cx, cy = center
        length = 10
        angles = {
            GridWorldEnv.UP: -math.pi / 2,
            GridWorldEnv.RIGHT: 0,
            GridWorldEnv.DOWN: math.pi / 2,
            GridWorldEnv.LEFT: math.pi,
        }
        ang = angles[action]
        ex = cx + int(math.cos(ang) * length)
        ey = cy + int(math.sin(ang) * length)
        sx = cx - int(math.cos(ang) * (length - 4))
        sy = cy - int(math.sin(ang) * (length - 4))

        arrow_color = (180, 200, 230)
        pygame.draw.line(self.screen, arrow_color, (sx, sy), (ex, ey), 2)
        # Punta de la flecha
        head_ang1 = ang + math.pi * 0.75
        head_ang2 = ang - math.pi * 0.75
        hx1 = ex + int(math.cos(head_ang1) * 5)
        hy1 = ey + int(math.sin(head_ang1) * 5)
        hx2 = ex + int(math.cos(head_ang2) * 5)
        hy2 = ey + int(math.sin(head_ang2) * 5)
        pygame.draw.polygon(self.screen, arrow_color, [(ex, ey), (hx1, hy1), (hx2, hy2)])

    def draw_panel(self):
        panel_rect = pygame.Rect(GRID_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, PANEL_BORDER, (GRID_WIDTH, 0), (GRID_WIDTH, WINDOW_HEIGHT), 2)

        px = GRID_WIDTH + 16
        y = 14

        # Título
        title = self.font_title.render("Laboratorio IA - Grid World", True, AGENT_COLOR)
        self.screen.blit(title, (px, y))
        y += 28

        # Métricas en vivo
        win_rate = (self.agent.wins / self.agent.episodes * 100) if self.agent.episodes > 0 else 0.0
        avg_reward = float(np.mean(self.agent.reward_history[-30:])) if self.agent.reward_history else 0.0

        stats = [
            f"Episodios: {self.agent.episodes}   |   Victorias: {self.agent.wins} ({win_rate:.0f}%)",
            f"Pasos Totales: {self.agent.total_steps}",
            f"Exploración (Epsilon): {self.agent.epsilon * 100:.1f}%",
            f"Recompensa Media (últ. 30): {avg_reward:.1f}",
        ]
        for line in stats:
            surf = self.font_body.render(line, True, TEXT_MAIN)
            self.screen.blit(surf, (px, y))
            y += 20

        # Sección Herramientas
        y = 126
        sub1 = self.font_sub.render("Click Izq: Poner | Click Der: Borrar", True, TEXT_MUTED)
        self.screen.blit(sub1, (px, y))

        # Sección Simulación
        sub2 = self.font_sub.render("Control de Entrenamiento:", True, TEXT_MUTED)
        self.screen.blit(sub2, (px, 224))

        # Dibujar todos los botones
        for btn in self.buttons:
            btn.draw(self.screen, self.font_small)

        # Gráfico de Recompensas en tiempo real
        chart_rect = pygame.Rect(px, 434, PANEL_WIDTH - 32, 130)
        self._draw_reward_chart(chart_rect)

        # Barra de estado inferior
        if self.status_timer > 0:
            self.status_timer -= 1
        status_surf = self.font_small.render(self.status_msg, True, ACCENT_GREEN)
        self.screen.blit(status_surf, (px, WINDOW_HEIGHT - 26))

    def _draw_reward_chart(self, rect: pygame.Rect):
        pygame.draw.rect(self.screen, GRID_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=6)

        label = self.font_small.render("Curva de Aprendizaje (Recompensa por Episodio)", True, TEXT_MUTED)
        self.screen.blit(label, (rect.x + 8, rect.y + 6))

        history = self.agent.reward_history
        if len(history) < 2:
            msg = self.font_small.render("Esperando completar episodios...", True, TEXT_MUTED)
            self.screen.blit(msg, msg.get_rect(center=rect.center))
            return

        plot_x = rect.x + 10
        plot_y = rect.y + 28
        plot_w = rect.width - 20
        plot_h = rect.height - 36

        min_r = min(min(history), -50.0)
        max_r = max(max(history), 100.0)
        span = max(max_r - min_r, 1.0)

        # Línea de cero
        if min_r < 0 < max_r:
            zero_y = plot_y + plot_h - int((0 - min_r) / span * plot_h)
            pygame.draw.line(self.screen, (60, 70, 90), (plot_x, zero_y), (plot_x + plot_w, zero_y), 1)

        points = []
        n = len(history)
        for i, val in enumerate(history):
            x = plot_x + int((i / (n - 1)) * plot_w)
            y = plot_y + plot_h - int(((val - min_r) / span) * plot_h)
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(self.screen, (99, 179, 237), False, points, 2)

    def run(self):
        running = True
        while running:
            _, steps_per_frame, target_fps = self.speed_modes[self.speed_idx]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Eventos de botones UI
                ui_clicked = False
                for btn in self.buttons:
                    if btn.handle_event(event):
                        ui_clicked = True

                if ui_clicked:
                    continue

                # Teclado rápido
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_pause()
                    elif event.key == pygame.K_r:
                        self.reset_brain()
                    elif event.key == pygame.K_c:
                        self.clear_map()
                    elif event.key == pygame.K_s:
                        self.save_all()
                    elif event.key == pygame.K_l:
                        self.load_all()
                    elif event.key == pygame.K_1:
                        self.tool_buttons[GridWorldEnv.WALL].callback()
                    elif event.key == pygame.K_2:
                        self.tool_buttons[GridWorldEnv.TRAP].callback()
                    elif event.key == pygame.K_3:
                        self.tool_buttons[GridWorldEnv.GOAL].callback()
                    elif event.key == pygame.K_4:
                        self.tool_buttons[GridWorldEnv.START].callback()

                # Dibujo en el mapa con el mouse
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.pos[0] < GRID_WIDTH:
                        self.mouse_drawing = True
                        self.erase_mode = (event.button == 3)
                        self.handle_grid_click(event.pos, right_click=self.erase_mode)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_drawing = False
                elif event.type == pygame.MOUSEMOTION and self.mouse_drawing:
                    self.handle_grid_click(event.pos, right_click=self.erase_mode)

            # Ejecutar pasos de entrenamiento de la IA
            if not self.paused:
                for _ in range(steps_per_frame):
                    self.step_simulation()

            # Renderizar interfaz
            self.screen.fill(BG_COLOR)
            self.draw_grid()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(target_fps)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = LabApp()
    app.run()
