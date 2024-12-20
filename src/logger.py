# Custom Logging Configuration
# This file sets up logging to:
# - Write log messages to a file in the "logs" directory.Dir name logs will be created and inside all logs file stored
# - Use a timestamped filename for each log file.
# - Format log messages with timestamp, line number, logger name, and level.
# - Log messages at INFO level and above (WARNING and ERROR will also be captured).
# Without this setup, logging defaults to console output with WARNING level only.


import logging
import os
from datetime import datetime

## instead of getting logger info display on console, we will create log file in same directory
name = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
file_name = f"{name}.log" 


## Creating working dir in which folder or file is saved
## location of file or path is current parent dir/ file_name

parent_directory = os.getcwd()
# creating path
directory_name = "logs"
directory_path = os.path.join(parent_directory, directory_name  )    #3 D:/.../logs

# creating this directory 

os.makedirs(directory_path, exist_ok = True)

# config logger file to get saved in this directory

file_path = os.path.join(directory_path, file_name)

# logg config to definf file path where to get saved and also defing its level
# you can initiate this logging on any script using any level after INFO or INFO()
logging.basicConfig(
    filename=file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# Example log message
logging.info("Logging has started.")





