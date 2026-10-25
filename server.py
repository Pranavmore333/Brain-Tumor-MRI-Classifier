"""
Production Server Entrypoint.
Runs the FastAPI DICOMweb & MONAI Label AI service.
"""

import uvicorn
import yaml
import os

if __name__ == "__main__":
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "8000"))

    print(f"[*] Starting Clinical Brain Tumor MRI AI Service on http://{host}:{port}")
    uvicorn.run("src.service.app:app", host=host, port=port, reload=False)
