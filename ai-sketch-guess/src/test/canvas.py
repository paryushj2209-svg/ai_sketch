"""Serve the browser doodle game and the existing PyTorch classifier."""

from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from res import DEFAULT_MODEL, DoodlePredictor


PAGE = Path(__file__).with_name("index.html")
MAX_REQUEST_BYTES = 4 * 1024 * 1024


def make_handler(predictor: DoodlePredictor) -> type[BaseHTTPRequestHandler]:
    class GameHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/assets/race-car-cursor.png":
                asset = PAGE.parent / "assets" / "race-car-cursor.png"
                try:
                    body = asset.read_bytes()
                except OSError as exc:
                    self._send_json(500, {"error": f"Could not load cursor image: {exc}"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(body)
                return
            if route == "/api/categories":
                self._send_json(200, {"categories": predictor.classes})
                return
            if route == "/":
                try:
                    body = PAGE.read_bytes()
                except OSError as exc:
                    self._send_json(500, {"error": f"Could not load game page: {exc}"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/predict":
                self._send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    self._send_json(413, {"error": "Drawing request is empty or too large"})
                    return
                payload = json.loads(self.rfile.read(length))
                target = payload["target"]
                if target not in predictor.classes:
                    self._send_json(400, {"error": "Unknown target category"})
                    return
                image_data = payload["image"].split(",", 1)[-1]
                image_bytes = base64.b64decode(image_data, validate=True)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    probabilities = dict(predictor.predict(image, top_k=len(predictor.classes)))
                target_probability = probabilities[target]
                guess, guess_probability = max(probabilities.items(), key=lambda item: item[1])
                print(
                    f"{datetime.now().strftime('%H:%M:%S')} | AI guessed {guess} "
                    f"({guess_probability:.1%}) | {target} score {target_probability:.1%}",
                    flush=True,
                )
                self._send_json(
                    200,
                    {
                        "target": target,
                        "score": target_probability,
                        "guess": guess,
                        "guess_probability": guess_probability,
                    },
                )
            except (KeyError, TypeError, ValueError, binascii.Error, OSError) as exc:
                self._send_json(400, {"error": f"Could not read drawing: {exc}"})
            except Exception as exc:  # Return inference errors to the browser instead of dropping the request.
                self._send_json(500, {"error": f"Prediction failed: {exc}"})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

    return GameHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        predictor = DoodlePredictor(args.model)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if len(predictor.classes) < 3:
        parser.error("The trained model needs at least three categories for the game")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(predictor))
    print(f"Doodle game ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping doodle game")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
