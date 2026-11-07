"""Check visible demonstration results through the production backends."""
from pathlib import Path
import subprocess
import re
import math
import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'build_c17/radius_demo.exe'
pytestmark = pytest.mark.skipif(not BIN.is_file(), reason='radius_demo is not built')

def run(method, k='8', tile='2'):
    return subprocess.run([str(BIN), method, str(ROOT/'examples/points.xyz'),
                           str(ROOT/'examples/queries.xyz'), k, tile],
                          capture_output=True, text=True, timeout=60)

@pytest.mark.parametrize('method', ['cpu-grid', 'cuda-grid', 'rt-incore', 'rt-ooc'])
def test_complete_demo(method):
    p = run(method)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'Q0 (0, 0, 0): count=4 IDs=[0, 1, 2, 4] truncated=no' in p.stdout
    assert 'Q1 (2, 0, 0): count=2 IDs=[1, 3] truncated=no' in p.stdout
    assert 'Q2 (10, 10, 10): count=0 IDs=[] truncated=no' in p.stdout
    assert 'PASS: all 3 query counts and returned IDs verified against brute force.' in p.stdout
    timing = re.search(r'Timing \(ms\): build=([0-9.]+) search=([0-9.]+) '
                       r'build\+search=([0-9.]+) verification=([0-9.]+)', p.stdout)
    assert timing, p.stdout
    build, search, total, verification = map(float, timing.groups())
    assert all(math.isfinite(x) and x >= 0 for x in (build, search, total, verification))
    assert total > 0 and abs(total - build - search) <= 0.000003
    memory = re.search(r'Memory \(bytes\): device=(\d+) pinned_host_tiles=(\d+)', p.stdout)
    assert memory, p.stdout
    device, pinned = map(int, memory.groups())
    if method == 'cpu-grid':
        assert device == pinned == 0
    elif method == 'cuda-grid':
        # Six 16-byte points, six links, 5^3 cell heads, three query/count/8-ID records.
        assert device == 6 * 20 + 125 * 4 + 3 * (16 + 4 * 9)
        assert pinned == 0
    else:
        assert device > 0 and pinned > 0

@pytest.mark.parametrize('method', ['cpu-grid', 'cuda-grid', 'rt-incore', 'rt-ooc'])
def test_truncation_preserves_counts(method):
    p = run(method, '1')
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'Q0 (0, 0, 0): count=4' in p.stdout
    assert 'Q1 (2, 0, 0): count=2' in p.stdout
    assert p.stdout.count('truncated=yes') == 2
    assert 'PASS:' in p.stdout
    assert 'rerun with max_ids at least 4' in p.stdout

def test_count_only():
    p = run('cpu-grid', '0')
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'count=4 IDs=[] truncated=yes' in p.stdout
    assert 'count=0 IDs=[] truncated=no' in p.stdout

def test_unknown_method():
    p = run('unknown')
    assert p.returncode != 0
    assert 'Unknown method' in p.stderr

def test_actual_tile_ownership_and_routing():
    p = run('rt-ooc')
    assert p.returncode == 0, p.stdout + p.stderr
    tiles = re.findall(r'Tile (\d+): IDs=\[([^]]*)\] points=(\d+)', p.stdout)
    assert [(int(t), ids, int(n)) for t, ids, n in tiles] == [
        (0, '0, 4', 2), (1, '1, 2', 2), (2, '3, 5', 2)]
    ids = [int(i) for _, members, _ in tiles for i in members.split(', ')]
    assert sorted(ids) == list(range(6))
    assert 'bounds: (0, 0, 0) -> (0, 0, 0)' in p.stdout
    assert 'bounds: (0, 0, 0) -> (1, 1, 0)' in p.stdout
    assert 'bounds: (2, 0, 0) -> (5, 5, 5)' in p.stdout
    assert 'Q0 candidate tiles=[0, 1]' in p.stdout
    assert 'Q1 candidate tiles=[1, 2]' in p.stdout
    assert 'Q2 candidate tiles=[]' in p.stdout

@pytest.mark.parametrize('tile,expected', [('1', 6), ('3', 2), ('6', 1)])
def test_tile_capacity_changes_partition_not_answers(tile, expected):
    p = run('rt-ooc', tile=tile)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f'RT tiles: {expected} |' in p.stdout
    assert 'Q0 (0, 0, 0): count=4 IDs=[0, 1, 2, 4] truncated=no' in p.stdout
    assert 'Q1 (2, 0, 0): count=2 IDs=[1, 3] truncated=no' in p.stdout
    assert 'Q2 (10, 10, 10): count=0 IDs=[] truncated=no' in p.stdout

def test_resident_index_remains_one_tile():
    p = run('rt-incore', tile='1')
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'RT tiles: 1 |' in p.stdout
    assert 'Tile 0: IDs=[0, 4, 1, 2, 3, 5] points=6' in p.stdout
    assert 'Q0 candidate tiles=[0]' in p.stdout
    assert 'Q2 candidate tiles=[]' in p.stdout

@pytest.mark.parametrize('tile', ['0', '-1', 'bad', '4294967296'])
def test_invalid_tile_capacity(tile):
    p = run('rt-ooc', tile=tile)
    assert p.returncode != 0 and 'tile_points' in p.stderr
