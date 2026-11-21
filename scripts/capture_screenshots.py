"""Capture verified high-resolution screenshots of all 5 Streamlit application pages via Chrome CDP."""

import asyncio
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path
import websockets

SCREENSHOT_DIR = Path("/Users/niyatijha/Desktop/crop-disease-detection/artifacts/figures/ui_screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("home", "home_overview.png"),
    ("disease", "disease_classification.png"),
    ("pest", "pest_damage_detection.png"),
    ("video", "video_surveillance.png"),
    ("analytics", "analytics_and_about.png"),
]


async def capture_all():
    # Start Chrome with remote debugging
    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless=new",
        "--remote-debugging-port=9222",
        "--user-data-dir=/tmp/chrome_cdp_profile",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1280,1024",
        "about:blank",
    ]
    proc = subprocess.Popen(chrome_cmd)
    time.sleep(2.5)

    try:
        # Get target tab websocket URL
        req = urllib.request.Request("http://127.0.0.1:9222/json/list")
        tabs = json.loads(urllib.request.urlopen(req).read().decode())
        ws_url = tabs[0]["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
            # Enable Page domain
            await ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
            await ws.recv()

            # Set viewport
            await ws.send(json.dumps({
                "id": 2,
                "method": "Emulation.setDeviceMetricsOverride",
                "params": {
                    "width": 1280,
                    "height": 1024,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                }
            }))
            await ws.recv()

            msg_id = 10
            for page_param, out_filename in PAGES:
                target_url = f"http://localhost:8501/?page={page_param}"
                print(f"Navigating to: {target_url}")

                msg_id += 1
                await ws.send(json.dumps({
                    "id": msg_id,
                    "method": "Page.navigate",
                    "params": {"url": target_url}
                }))
                await ws.recv()

                # Allow Streamlit websocket and UI to fully render
                await asyncio.sleep(4.0)

                # Capture full page screenshot
                msg_id += 1
                await ws.send(json.dumps({
                    "id": msg_id,
                    "method": "Page.captureScreenshot",
                    "params": {"format": "png"}
                }))

                # Read responses until we get the screenshot result
                while True:
                    resp_raw = await ws.recv()
                    resp = json.loads(resp_raw)
                    if resp.get("id") == msg_id:
                        img_bytes = base64.b64decode(resp["result"]["data"])
                        out_path = SCREENSHOT_DIR / out_filename
                        out_path.write_bytes(img_bytes)
                        print(f"Saved {out_filename} ({len(img_bytes)} bytes)")
                        break

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    asyncio.run(capture_all())
