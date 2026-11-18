"""Move result runs to stage-first names and update live path references."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[3]
base = root / 'results/planar_residual_depth'
mapping = {}
for path in sorted(base.iterdir()):
    name = path.name
    if not path.is_dir() or re.match(r'^P\d', name):
        continue
    if name.startswith('indoor-'):
        name_new = 'P10-' + name
    elif name == 'p1-downloads':
        name_new = 'P1-downloads'
    else:
        match = re.fullmatch(r'(.*?)-(P\d[^-]*(?:-P\d[^-]*)?)-(\d{8}-\d{3})', name)
        if not match:
            continue
        sample, stage, suffix = match.groups()
        name_new = f'{stage}-{sample}-{suffix}'
    mapping[name] = name_new
assert len(set(mapping.values())) == len(mapping)
assert all(not (base / target).exists() for target in mapping.values())
if not mapping:
    raise SystemExit('No further moves.')
pattern = re.compile('|'.join(map(re.escape, sorted(mapping, key=len, reverse=True))))
# Keep archived code snapshots and hash inventories intact. Record their original paths in the map.
updates = []
for folder in [root/'experiments', root/'docs', base]:
    for path in folder.rglob('*'):
        if not path.is_file() or path.suffix not in {'.py','.md','.json','.sh','.csv','.txt','.yaml'}:
            continue
        if path == Path(__file__).resolve():
            continue
        rel = path.relative_to(folder)
        if folder == base and (any(part in {'source','dependency_audit','archive_first_render_attempt'} for part in rel.parts) or path.name.startswith('fit_hashes')):
            continue
        old = path.read_text(errors='strict')
        new = pattern.sub(lambda m: mapping[m[0]], old)
        if new != old:
            updates.append((path,new))
for path, content in updates:
    path.write_text(content)
for old, new in mapping.items():
    (base/old).rename(base/new)
(base/'result_directory_rename_map.json').write_text(json.dumps(mapping,indent=2)+'\n')
print(f'{len(mapping)} directories, {len(updates)} updated text files')
