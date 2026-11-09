"""Verify the delivered replacement and that inherited project files are unchanged."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'source_replay_install_manifest.json').read_text(encoding='utf-8'))
failures = []
for relative, expected in manifest['expected_files'].items():
    path = root / relative
    if not path.is_file():
        failures.append({'file': relative, 'problem': 'missing'})
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append({'file': relative, 'problem': 'modified'})
print(json.dumps({'variant': manifest['variant'], 'checked_files': len(manifest['expected_files']),
                  'failures': failures, 'verified': not failures}, indent=2))
sys.exit(int(bool(failures)))
