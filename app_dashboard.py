import logging
from dashboard.app import app as dashboard_app

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Start Dashboard!")
    dashboard_app.run_server(host="127.0.0.1", port=8888, debug=True)
