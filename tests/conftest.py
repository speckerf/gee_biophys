import ee
import pytest


@pytest.fixture(scope="session")
def ee_init():
    """Initialize Earth Engine for testing."""
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()
