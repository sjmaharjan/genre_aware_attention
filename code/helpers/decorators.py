import codecs
from contextlib import contextmanager
import sys
from functools import wraps

# A lazy decorator
def lazy(func):
    """ A decorator function designed to wrap attributes that need to be
        generated, but will not change. This is useful if the attribute is
        used a lot, but also often never used, as it gives us speed in both
        situations.

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        name = "_" + func.__name__
        try:
            return getattr(self, name)
        except AttributeError:
            value = func(self, *args, **kwargs)
            setattr(self, name, value)
            return value

    return wrapper






# Ref :Raymond Hettinger beautiful python code slides
# Python 3 has this features but python 2 does not
@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout