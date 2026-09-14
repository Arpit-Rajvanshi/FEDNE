import sys

# Polyfill for Python 3.11 pre-release compatibility with PyTorch/torchvision dynamo
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits = lambda maxdigits: None

__version__ = "0.1.0"
