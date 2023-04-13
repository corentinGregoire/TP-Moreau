import os
import logging
import uvicorn

from app.api import app


if __name__ == '__main__':
    logging.info("starting uvicorn web server")
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv("PORT", 8000)), log_level='info')
