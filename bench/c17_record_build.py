"""Record the just-built C17 source/executable identity; invoked after a successful build."""
import argparse,json,subprocess
from pathlib import Path
from c17_evidence import ROOT,digest,source_tree_hash
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',default='build_c17');a=p.parse_args();b=ROOT/a.build
 compiler=next(b.glob('CMakeFiles/*/CMakeCCompiler.cmake')).read_text()
 metadata=dict(status='built',binary_sha256=digest(b/'ooc_rtnn.exe'),source_tree_sha256=source_tree_hash(),configuration='Release',compiler=compiler,cuda=subprocess.check_output(['nvcc','--version'],text=True),optix_revision=subprocess.check_output(['git','-C',str(b/'_deps/optix_dev-src'),'rev-parse','HEAD'],text=True).strip())
 (b/'build_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n');print('C17 build provenance recorded')
if __name__=='__main__':main()
