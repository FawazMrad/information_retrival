# ============================================================
# run_services.py  -  start all SOA services (one per process)
#
#   python run_services.py
#
# Each service can also be started on its own, e.g.:
#   uvicorn api.retrieval_service:app --port 8004
# ============================================================
import sys
import time
import subprocess

import config

SERVICES = [
    ("preprocessing", "api.preprocessing_service"),
    ("indexing", "api.indexing_service"),
    ("refinement", "api.refinement_service"),
    ("retrieval", "api.retrieval_service"),
    ("evaluation", "api.evaluation_service"),
    ("gateway", "api.gateway"),
]


def main():
    procs = []
    print("Starting services (Ctrl+C to stop all)...\n")
    for name, module in SERVICES:
        port = config.SERVICES[name]
        p = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", f"{module}:app",
             "--host", config.SERVICE_HOST, "--port", str(port)])
        procs.append((name, p))
        print(f"  {name:<14} -> http://{config.SERVICE_HOST}:{port}")
        time.sleep(1)

    print("\nAll services launching. Heavy ones (retrieval) take ~30-60s to load.")
    print("Gateway docs:  http://127.0.0.1:8000/docs")
    try:
        for _, p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all services ...")
        for _, p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
