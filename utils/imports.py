'''
Dynamic import 
'''
from typing import Callable
import sys
import traceback

def import_class(import_str : str):
    '''
    Imports the a module 
    Return: 
        Object 
    '''
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try: 
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))