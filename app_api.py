import logging
import uvicorn

from api.app import app as api_app

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Start API!")
    uvicorn.run(api_app, host="127.0.0.1", port=8889, debug=True)
