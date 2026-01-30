import logging
from datetime import datetime
import os

# 1. Pure filename
LOGFILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Directory for logs (no filename here)
logs_dir = os.path.join(os.getcwd(), "log")

# 3. Ensure directory exists
os.makedirs(logs_dir, exist_ok=True)

# 4. Full file path inside that directory
LOG_FILE_PATH = os.path.join(logs_dir, LOGFILE)

# 5. Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
