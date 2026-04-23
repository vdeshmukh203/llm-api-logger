import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def test_import():
    import llm_api_logger as lal
    assert hasattr(lal, 'LogRecord')

def test_backends():
    import llm_api_logger as lal
    assert hasattr(lal, 'JSONLBackend')
    assert hasattr(lal, 'SQLiteBackend')
    assert hasattr(lal, 'StdoutBackend')

def test_detect_provider():
    import llm_api_logger as lal
    assert callable(lal._detect_provider)

def test_extract_model():
    import llm_api_logger as lal
    assert callable(lal._extract_model)
