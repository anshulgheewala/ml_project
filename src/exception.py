import sys

def error_message_details(error, error_detail: sys):
    # on which file and line error has occoured will be stored in the below variable
    _,_,exec_tb=error_detail.exc_info()
    file_name = exec_tb.tb_frame.f_code.co_filename
    error_message="Error occoured in python script name [{0}] line number [{1}] error message [{2}]".format(

        file_name, exec_tb.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message