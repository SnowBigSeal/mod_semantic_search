import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
import config

if __name__ == "__main__":
    cfg = config.load()
    host = cfg.get("web", {}).get("host", "127.0.0.1")
    port = cfg.get("web", {}).get("port", 8000)
    print(f"Starting Mod Search at http://{host}:{port}")
    uvicorn.run("web.app:app", host=host, port=port, reload=True)
