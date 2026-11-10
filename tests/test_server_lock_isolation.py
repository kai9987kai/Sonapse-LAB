"""Runtime-independent endpoints must not block behind a long runtime.step()."""
import threading
import time

from synapse.server import Workbench


class _Cfg:
    seed = 42


class _StubRuntime:
    config = _Cfg()
    transition_taps = []


def test_aux_endpoints_respond_while_runtime_lock_held():
    bench = Workbench(runtime=_StubRuntime())
    held = threading.Event()
    release = threading.Event()

    def hog():
        with bench.lock:  # simulate a slow runtime.step() holding the runtime lock
            held.set()
            release.wait(10)

    t = threading.Thread(target=hog, daemon=True)
    t.start()
    assert held.wait(5)
    try:
        t0 = time.perf_counter()
        assert bench.get('/api/ecology/soundscape')['status'] == 'active'
        assert bench.get('/api/connectome/fusion/summary')['total_neurons'] == 12082
        assert 'status' in bench.get('/api/ecology/tournament/status')
        assert bench.post('/api/ecology/vocalize', {'niche': 'forager'})['status'] == 'emitted'
        assert bench.post('/api/ecology/step', {'steps': 2})['status'] == 'completed'
        assert time.perf_counter() - t0 < 5.0
    finally:
        release.set()
        t.join(5)


def test_runtime_endpoints_still_serialised_by_runtime_lock():
    """Endpoints that touch the runtime keep the original lock discipline."""
    bench = Workbench(runtime=_StubRuntime())
    held = threading.Event()
    release = threading.Event()
    result = {}

    def hog():
        with bench.lock:
            held.set()
            release.wait(10)

    def consolidation():
        result['v'] = bench.get('/api/sleep/consolidation/status')

    t = threading.Thread(target=hog, daemon=True)
    t.start()
    assert held.wait(5)
    c = threading.Thread(target=consolidation, daemon=True)
    c.start()
    c.join(0.5)
    assert c.is_alive()  # blocked behind the runtime lock, as before
    release.set()
    c.join(5)
    t.join(5)
    assert 'v' in result
