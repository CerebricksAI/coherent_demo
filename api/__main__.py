import os

import uvicorn

if __name__ == "__main__":
    # Azure injects PORT; local dev can use API_PORT in .env (default 8001)
    port = int(os.getenv("PORT") or os.getenv("API_PORT", "8001"))
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=port,
        reload=os.getenv("API_RELOAD", "false").lower() in ("1", "true", "yes"),
    )
