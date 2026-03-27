"""
Digital Twin Launcher
======================
Single entry point that initialises the ROM engine and launches the
interactive Dash dashboard in the browser.

Usage::

    python run_digital_twin.py                 # launch dashboard
    python run_digital_twin.py --api           # launch REST API instead
"""
#%%
import os
import sys
import shutil
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – identical pattern to RL_Refactored/utilities/__init__.py
# ---------------------------------------------------------------------------
try:
    _THIS_DIR = Path(__file__).resolve().parent       # DigitalTwin/
except NameError:
    _THIS_DIR = Path(os.getcwd())
    if _THIS_DIR.name != "DigitalTwin":
        _THIS_DIR = _THIS_DIR / "DigitalTwin"
_PROJECT_ROOT = _THIS_DIR.parent                      # ROM-Optimization/

_ROM_DIR = _PROJECT_ROOT / "ROM_Refactored"
_RL_DIR = _PROJECT_ROOT / "RL_Refactored"
# ROM_Refactored must be earliest so its internal `from utilities...` imports resolve.
# insert(0) reverses order, so we iterate in reverse priority.
for p in [str(_PROJECT_ROOT), str(_RL_DIR), str(_ROM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml


def main():
    parser = argparse.ArgumentParser(description="Reservoir Digital Twin")
    parser.add_argument("--config", default=str(_THIS_DIR / "config.yaml"),
                        help="Path to Digital Twin config YAML")
    parser.add_argument("--weights", default=None,
                        help="Path to ROM weights (.pth/.pt)")
    parser.add_argument("--api", action="store_true",
                        help="Launch the FastAPI REST server instead of the dashboard")
    parser.add_argument("--port", type=int, default=None,
                        help="Override the default port")
    args, _ = parser.parse_known_args()

    # ---- Clean up stale caches ----
    _SKIP = {"kit-app-template", "omniverse", "node_modules", ".git"}
    for root, dirs, _ in os.walk(str(_THIS_DIR)):
        dirs[:] = [d for d in dirs if d not in _SKIP]
        if Path(root).name == "__pycache__":
            shutil.rmtree(root, ignore_errors=True)
            dirs.clear()
    exports_dir = _THIS_DIR / "exports"
    if exports_dir.exists():
        shutil.rmtree(exports_dir, ignore_errors=True)

    # ---- Load DT config ----
    with open(args.config, "r") as f:
        dt_cfg = yaml.safe_load(f)

    # ---- Build engine ----
    from DigitalTwin.engine import DigitalTwinEngine

    print("=" * 60)
    print("  Reservoir Digital Twin")
    print("=" * 60)
    engine = DigitalTwinEngine(
        dt_config_path=args.config,
        rom_weights_path=args.weights,
    )
    engine.reset()
    print(f"  Grid : {engine.nx} x {engine.ny} x {engine.nz}")
    print(f"  Device: {engine.device}")
    print("=" * 60)

    # ---- Mode: REST API ----
    if args.api:
        import uvicorn
        from DigitalTwin.services import create_api_app

        api_app = create_api_app(engine)
        host = dt_cfg.get("services", {}).get("host", "127.0.0.1")
        port = args.port or dt_cfg.get("services", {}).get("port", 8051)
        print(f"  API server -> http://{host}:{port}/docs")
        uvicorn.run(api_app, host=host, port=port)
        return

    # ---- Mode: Dashboard (default) ----
    from DigitalTwin.dashboard import create_dash_app
    import socket

    app = create_dash_app(engine, dt_cfg)
    host = dt_cfg.get("dashboard", {}).get("host", "127.0.0.1")
    port = args.port or dt_cfg.get("dashboard", {}).get("port", 8050)
    debug = dt_cfg.get("dashboard", {}).get("debug", True)

    def _port_free(h, p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((h, p)) != 0

    if not _port_free(host, port):
        for p in range(port + 1, port + 20):
            if _port_free(host, p):
                print(f"  Port {port} in use, switching to {p}")
                port = p
                break

    print(f"  Dashboard -> http://{host}:{port}")
    print("  Press Ctrl+C to stop.")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

# %%
