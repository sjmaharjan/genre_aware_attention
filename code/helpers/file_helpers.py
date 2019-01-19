from contextlib import contextmanager
import sys

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