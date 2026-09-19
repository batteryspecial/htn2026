"""Drive a running perception service the way the orchestrator will.

    python scripts/e2e_check.py [base_url]

Sends two instructions over HTTP, listens on both websockets, and prints the
retask latency for each: instruction accepted -> phase TRACKING. That number is
what the demo is judged on.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx
import websockets

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"
WS = BASE.replace("http", "ws")

INSTRUCTIONS = [
    ("i1", {"spec_id": "s1", "model": "yoloe", "mode": "follow",
            "targets": [{"ref": "t1", "detect": ["person"], "select": "largest"}]}),
    ("i2", {"spec_id": "s2", "model": "yoloe", "mode": "center",
            "targets": [{"ref": "t1", "detect": ["bus"], "select": "most_centered"}]}),
]


async def watch(url, label, sink, stop):
    async with websockets.connect(url) as ws:
        while not stop.is_set():
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 0.4))
            except (TimeoutError, asyncio.TimeoutError):
                continue
            sink.append((time.time(), label, msg))


async def main() -> int:
    stop = asyncio.Event()
    events, targets = [], []
    tasks = [
        asyncio.create_task(watch(f"{WS}/ws/events", "event", events, stop)),
        asyncio.create_task(watch(f"{WS}/ws/target", "target", targets, stop)),
    ]
    await asyncio.sleep(0.5)

    ok = True
    async with httpx.AsyncClient(base_url=BASE, timeout=10) as http:
        for instruction_id, spec in INSTRUCTIONS:
            detect = spec["targets"][0]["detect"][0]
            print(f'\n> "follow the {detect}"')
            sent = time.time()
            r = await http.post("/spec", json={"instruction_id": instruction_id, "spec": spec})
            print(f"  POST /spec -> {r.status_code} ({1000*(time.time()-sent):.0f} ms)")
            if r.status_code != 202:
                print(f"  rejected: {r.text}")
                ok = False
                continue

            tracking = None
            deadline = time.time() + 8
            while time.time() < deadline:
                for ts, _, m in targets:
                    if ts > sent and m.get("phase") == "TRACKING" and m.get("spec_id") == spec["spec_id"]:
                        tracking = ts
                        break
                if tracking:
                    break
                await asyncio.sleep(0.02)

            stages = [m["stage"] for ts, lbl, m in events if lbl == "event" and ts > sent]
            print(f"  events: {stages}")
            if tracking:
                print(f"  RETASK LATENCY: {1000*(tracking-sent):.0f} ms")
                last = [m for _, lbl, m in targets if lbl == "target" and m.get("visible")][-1]
                print(f"  target: {last['label']} cx={last['cx']:+.2f} cy={last['cy']:+.2f} "
                      f"area={last['area']:.3f} conf={last['conf']:.2f} track={last['track_id']}")
            else:
                print("  never reached TRACKING (nothing matched the prompt)")
                ok = False
            await asyncio.sleep(1.0)

        h = (await http.get("/health")).json()
        print(f"\nhealth: phase={h['phase']} fps={h['fps']} model={h['model']} "
              f"timings={h['timings_ms']}")
        r = await http.get("/frame.jpg")
        print(f"frame.jpg: {r.status_code} {len(r.content)} bytes jpeg={r.content[:2] == bytes([255, 216])}")

        print("\nnegative checks:")
        r = await http.post("/spec", json={"instruction_id": "bad1", "spec": {
            "spec_id": "sx", "model": "rfdetr", "targets": [{"ref": "t", "detect": ["cat"]}]}})
        print(f"  unknown model      -> {r.status_code} {r.json().get('detail','')[:60]}")
        r = await http.post("/spec", json={"instruction_id": "bad2", "spec": {
            "spec_id": "sy", "model": "coco", "targets": [{"ref": "t", "detect": ["pencil"]}]}})
        print(f"  coco has no pencil -> {r.status_code} {r.json().get('detail','')[:60]}")
        r = await http.delete("/spec")
        print(f"  DELETE /spec       -> {r.status_code}")
        await asyncio.sleep(0.5)
        print(f"  phase after delete -> {(await http.get('/health')).json()['phase']}")

    stop.set()
    for t in tasks:
        t.cancel()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
