"""Download Stanford Armadillo mirror and retain exact coordinates in supported PLY."""
import hashlib, json, urllib.request
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

URL = "https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/ArmadilloMesh.ply"

def prepare():
    target = ROOT / "data/stanford/Armadillo.ply"
    provenance = RESULTS / "armadillo_provenance.json"
    if target.exists() and provenance.exists():
        metadata = json.loads(provenance.read_text())
        if hashlib.sha256(target.read_bytes()).hexdigest() == metadata["vertex_sha256"]:
            return target
        raise ValueError("Armadillo checksum mismatch")
    content = urllib.request.urlopen(URL, timeout=60).read()
    offset = content.index(b"end_header\n") + len(b"end_header\n")
    header = content[:offset].decode()
    if "format binary_big_endian 1.0" not in header: raise ValueError("unexpected PLY encoding")
    count = int(next(line.split()[-1] for line in header.splitlines() if line.startswith("element vertex")))
    if header.split("element vertex")[1].split("element face")[0].splitlines()[1:] != ["property float x", "property float y", "property float z"]:
        raise ValueError("unexpected vertex layout")
    vertices = np.frombuffer(content, dtype=">f4", count=count*3, offset=offset).astype("<f4")
    output = f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nend_header\n".encode() + vertices.tobytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(output)
    metadata = dict(url=URL, source_sha256=hashlib.sha256(content).hexdigest(), vertex_sha256=hashlib.sha256(output).hexdigest(),
                    vertices=count, preprocessing="Endian conversion only; faces omitted; coordinates unchanged", units="Original coordinates, no rescaling",
                    credit="Stanford Computer Graphics Laboratory", terms="https://graphics.stanford.edu/data/3Dscanrep/")
    provenance.write_text(json.dumps(metadata, indent=2))
    return target

if __name__ == "__main__": print(prepare())
