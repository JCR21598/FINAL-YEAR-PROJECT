#
#   Decorators for the Application, to extend the functionality and visualsation of some functions
#


def print_simple_divider(original_func):
    def wrapper(*args, **kwargs):
        print("\n\n____________________________________________________________________________________________")
        return original_func(*args, **kwargs)
    return wrapper

