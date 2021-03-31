import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip_sefr", action="store_true", default=False, help="skip tests requiring SEFR tokenizer"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "sefr: mark test that it requires SEFR tokenizer to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip_sefr"):
       
        skip_sefr = pytest.mark.skip(reason="skip tests requiring SEFR tokenizer")
        for item in items:
            if "sefr" in item.keywords:
                item.add_marker(skip_sefr)