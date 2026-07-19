import sys

# PyTorch dynamo compatibility monkeypatch for python version 3.11 environment
if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int:
        return 4300
    sys.get_int_max_str_digits = get_int_max_str_digits
if not hasattr(sys, "set_int_max_str_digits"):
    def set_int_max_str_digits(maxdigits: int) -> None:
        pass
    sys.set_int_max_str_digits = set_int_max_str_digits
