## sys module contain various functionsand has acess to SYSTEM-SPECIFIC parameters 
## and function that interact with python run time enviroment. It mange i/o, error handling/other syst related features
## whever in script u use try catch, and under catch if u raise exceptions, it will return error message in detail

# Purpose:

# This function is designed to capture and return a custom error message 
# whenever an exception is raised. whenever in script   try catch used , and under catch if it raises exceptions, 
# it will return error message in detail.
#The message includes details such as 
# the name of the script, the line number where the error occurred, and the actual 
# error message raised from the exception.
#
# Parameters:
# - error: The exception or error message that was raised.
# - error_detail: The sys module, which is used to access the exc_info() function 
#   to gather traceback information.
#
# Returns:
# - A detailed string message indicating where the error occurred (script name, line number),
#   and what caused the error (the exception message passed from the raised exception).


# it helps to generate more detailed message from error u get from exception class
import sys
from src.logger import logging
#import logging

def error_message_detail(error, error_detail:sys):

    # exc_tb contains traceback obj value
    _, _, exc_tb = error_detail.exc_info()
    # getting filename from traceback obj
    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    # error message will be returned as
    error_message = "Error occured in python script name [{0}] at line number [{1}] error message [{2}]".format(filename, lineno, str(error))
    return error_message
    



# This CustomException class inherits the Exception class in python.
# Using super() helps to invoke _init__  method of Exception  class 
# The Exception class has it own __int__ which stores or process error messages
# which is used in CustomException class as error message 


# purpose: This class inherits from Python’s built-in Exception class and is designed to 
# create a custom exception that includes additional details about the error location and 
# message using the error_message_detail function.
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):

        # from parent class Exception you get error_message from catch 
        super().__init__(error_message)
        
        # basic error exception message is then  used to provide more detailed error message
        # using exc_info() 
        # error_message is value u get returned from def error_message_detail ie line no , ..
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

# if __name__ == "__main__":
#     try: 
#        a = 1/0
#     except Exception as e :
#         logging.info("exception") # this will get logged in logged file
#         raise CustomException(e, sys)


