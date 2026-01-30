import sys
import traceback

def exception_error_handling(error, error_detail=None):
    if error_detail is None:
        error_detail = sys.exc_info()
    
    try:
        _, _, exc_tb = error_detail
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        return f"Error in [{file_name}:{line_no}]: {str(error)}"
    except:
        return f"Error: {str(error)}"  # Fallback

class Custom_Exception(Exception):
    def __init__(self, error, error_detail=None):  # ✅ Default None
        super().__init__(str(error))
        self.error_message = exception_error_handling(error, error_detail)

