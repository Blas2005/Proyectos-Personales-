import asyncio
import base64
import hashlib
import json
import struct
from pathlib import Path

WS_MAGIC_STRING = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebVisualizerBridge:
    """
    Servidor híbrido HTTP + WebSocket (RFC 6455) usando únicamente la librería estándar de Python.
    Sirve el visor 3D (Three.js) en http://localhost:8000 y sincroniza en tiempo real
    el estado del auto, los sensores láser y las neuronas de la red por WebSocket.
    """

    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.clients = set()
        self.sim_speed = 1  # 1x por defecto, modificable desde la UI web (1x, 5x, 25x)
        self.paused = False
        self.reset_requested = False
        self.web_dir = Path(__file__).parent / "web"

    async def start(self):
        server = await asyncio.start_server(self._handle_client, self.host, self.port)
        print(f"============================================================")
        print(f" 🌐 Visor 3D activo en: http://{self.host}:{self.port}")
        print(f"============================================================")
        return server

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                return

            parts = request_line.decode("utf-8", errors="ignore").strip().split(" ")
            path = parts[1] if len(parts) > 1 else "/"

            headers = {}
            while True:
                line = await reader.readline()
                if not line or line == b"\r\n":
                    break
                decoded = line.decode("utf-8", errors="ignore").strip()
                if ":" in decoded:
                    k, v = decoded.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            # Si es una conexión WebSocket
            if headers.get("upgrade", "").lower() == "websocket":
                await self._handle_websocket(reader, writer, headers)
            else:
                await self._serve_http(writer, path)
        except Exception:
            try:
                writer.close()
            except Exception:
                pass

    async def _serve_http(self, writer: asyncio.StreamWriter, path: str):
        html_file = self.web_dir / "index.html"
        if path in ("/", "/index.html") and html_file.exists():
            body = html_file.read_bytes()
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/html; charset=utf-8\r\n"
                b"Cache-Control: no-cache\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                + body
            )
        else:
            body = b"Not Found"
            response = (
                b"HTTP/1.1 404 Not Found\r\n"
                b"Content-Type: text/plain\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                + body
            )
        writer.write(response)
        await writer.drain()
        writer.close()

    async def _handle_websocket(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict):
        key = headers.get("sec-websocket-key", "")
        accept_raw = hashlib.sha1((key + WS_MAGIC_STRING).encode("utf-8")).digest()
        accept_key = base64.b64encode(accept_raw).decode("utf-8")

        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key}\r\n\r\n"
        )
        writer.write(handshake.encode("utf-8"))
        await writer.drain()

        self.clients.add(writer)
        try:
            while True:
                msg = await self._read_ws_frame(reader)
                if msg is None:
                    break
                self._on_client_message(msg)
        except Exception:
            pass
        finally:
            self.clients.discard(writer)
            try:
                writer.close()
            except Exception:
                pass

    async def _read_ws_frame(self, reader: asyncio.StreamReader):
        header = await reader.readexactly(2)
        b1, b2 = header[0], header[1]
        opcode = b1 & 0x0F
        if opcode == 0x8:  # Close frame
            return None
        masked = b2 & 0x80
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack(">H", await reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", await reader.readexactly(8))[0]

        mask_key = await reader.readexactly(4) if masked else None
        payload = bytearray(await reader.readexactly(length))
        if masked and mask_key:
            for i in range(length):
                payload[i] ^= mask_key[i % 4]
        return payload.decode("utf-8", errors="ignore")

    def _on_client_message(self, raw_msg: str):
        try:
            data = json.loads(raw_msg)
            cmd = data.get("type")
            if cmd == "set_speed":
                self.sim_speed = int(data.get("speed", 1))
            elif cmd == "toggle_pause":
                self.paused = not self.paused
            elif cmd == "reset_episode":
                self.reset_requested = True
        except Exception:
            pass

    async def broadcast(self, payload: dict):
        if not self.clients:
            return
        raw = json.dumps(payload).encode("utf-8")
        frame = bytearray([0x81])  # Fin + Text frame
        length = len(raw)
        if length < 126:
            frame.append(length)
        elif length < 65536:
            frame.append(126)
            frame.extend(struct.pack(">H", length))
        else:
            frame.append(127)
            frame.extend(struct.pack(">Q", length))
        frame.extend(raw)

        dead_clients = []
        for writer in list(self.clients):
            try:
                writer.write(frame)
                await writer.drain()
            except Exception:
                dead_clients.append(writer)
        for w in dead_clients:
            self.clients.discard(w)
