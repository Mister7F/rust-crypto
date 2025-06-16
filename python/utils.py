from contextlib import contextmanager


@contextmanager
def assert_raises(error_type, error_message):
    try:
        yield
    except Exception as e:
        assert isinstance(e, error_type)
        assert str(e) == error_message, f"Wrong error: {e}"
        return

    raise Exception("Test failed")
