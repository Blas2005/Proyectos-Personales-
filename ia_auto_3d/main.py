import asyncio
from env_simulator import Auto3DEnv
from cerebro_numpy import RedNeuronalRL
from server_bridge import WebVisualizerBridge


async def loop_entrenamiento():
    env = Auto3DEnv()
    cerebro = RedNeuronalRL(n_inputs=5, n_hidden=16, n_actions=3, learning_rate=0.06, gamma=0.82)
    bridge = WebVisualizerBridge(host="127.0.0.1", port=8000)

    await bridge.start()
    print("🚗 Entrenando al Auto 3D con Reinforcement Learning (NumPy puro)...")
    print("👉 Abrí http://127.0.0.1:8000 en tu navegador para ver el mundo 3D y la red neuronal en vivo.\n")

    episodio = 1
    mejor_recompensa = -999.0
    historial_recompensas = []

    while True:
        estado = env.reset()
        terminado = False
        ultimo_a1 = [0.0] * 8
        ultimas_probs = [0.33, 0.34, 0.33]
        ultima_accion = 1

        while not terminado:
            # Ejecutamos varios pasos por frame si el usuario eligió Turbo 5x o 25x en la web
            pasos_por_frame = max(1, bridge.sim_speed)

            for _ in range(pasos_por_frame):
                accion, a1, probs = cerebro.elegir_accion(estado)
                nuevo_estado, recompensa, terminado, _ = env.step(accion)
                cerebro.registrar_recompensa(recompensa)

                estado = nuevo_estado
                ultimo_a1 = [round(float(v), 3) for v in a1[:8]]
                ultimas_probs = [round(float(p), 3) for p in probs]
                ultima_accion = accion

                if terminado:
                    break

            # Enviamos el estado al visor 3D Web
            await bridge.broadcast({
                "x": round(env.x, 2),
                "z": round(env.z, 2),
                "heading": round(env.heading, 3),
                "sensors": [round(float(s), 3) for s in env.sensor_readings],
                "hits": env.sensor_hits,
                "hidden": ultimo_a1,
                "probs": ultimas_probs,
                "action": ultima_accion,
                "episode": episodio,
                "reward": round(env.total_reward, 1),
                "best_reward": round(max(mejor_recompensa, env.total_reward), 1),
                "steps": env.steps,
                "history": historial_recompensas[-60:],
            })

            await asyncio.sleep(0.025)  # ~40 FPS de transmisión visual

        # Al terminar el episodio (choque o vuelta completada), ejecutamos Backpropagation
        recompensa_ep = round(env.total_reward, 1)
        if recompensa_ep > mejor_recompensa:
            mejor_recompensa = recompensa_ep

        historial_recompensas.append(recompensa_ep)
        cerebro.entrenar_episodio()

        if episodio % 5 == 0 or episodio == 1:
            print(
                f"Episodio {episodio:03d} | Pasos: {env.steps:03d} | "
                f"Recompensa: {recompensa_ep:6.1f} | Mejor Récord: {mejor_recompensa:6.1f}"
            )

        episodio += 1


if __name__ == "__main__":
    try:
        asyncio.run(loop_entrenamiento())
    except KeyboardInterrupt:
        print("\nEntrenamiento detenido por el usuario.")
