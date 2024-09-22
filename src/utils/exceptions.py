import sys
import os
from utils.logger import logging

def generate_error_message(error_msg):
    _, _, exc_tb = sys.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    function_name = exc_tb.tb_frame.f_code.co_name
    message = "Error information:\nFilename: [{0}]\nFunction name: [{1}]\nLine number: [{2}]\nError: [{3}]".format(
        filename, function_name, lineno, error_msg)
    
    return message

class CustomException(Exception):
    def __init__(self, error_msg):
        self.error_msg = generate_error_message(error_msg)
        super().__init__(self.error_msg)
    
    def __str__(self):
        return self.error_msg
    

# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.info('Division by zero error')
#         raise CustomException(e)