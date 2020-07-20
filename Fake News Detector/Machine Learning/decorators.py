#
#       Author:     Juan Camilo Rodriguez
#
#   About File:     This file is for decorators to be able to extend and enhance the functionality of functions/methods
#                   in other files.
#

"""===     IMPORTS     ==="""

'''Third-party Imports'''


'''In-built Imports'''
from functools import wraps     # https://www.blog.pythonlibrary.org/2016/02/17/python-functools-wraps/
import timeit

'''Personal Imports'''




def section_divider(original_func):
    def wrapper(*args, **kwargs):

        colour = fg("magenta")
        res = attr("bold")

        print(colour + "\n\n"
              "-------------------------------------------------------------------------------------------------\n"
              "=================================================================================================\n"
              "-------------------------------------------------------------------------------------------------\n" + res)
        return original_func(*args, **kwargs)
    return wrapper


def simple_divider(original_func):
    def wrapper(*args, **kwargs):
        print("\n\n"
              "-------------------------------------------------------------------------------------------------\n")
        return original_func(*args, **kwargs)
    return wrapper


def timer(original_func):

    @wraps(original_func)
    def wrapper(*args, **kwargs):

        start_time = time.time()

        original_func()

        elapsed_time = time.time() - start_time

    return wrapper