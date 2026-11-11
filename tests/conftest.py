import pytest

from synapse.connectome_assets import get_source_inventory


@pytest.fixture
def require_connectome_sources():
    """Skip source-backed assays only when their pinned local data is absent."""
    inventory = {item['id']: item['availability'] for item in get_source_inventory()}

    def require(*datasets):
        missing = [name for name in datasets if inventory.get(name) not in ('prepared', 'raw_available')]
        if missing:
            pytest.skip('verified connectome source data unavailable: ' + ', '.join(missing))

    return require
