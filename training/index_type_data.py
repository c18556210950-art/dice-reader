"""Index and exact-deduplicate the type-only archive without fabricating face labels."""
from collections import Counter
import hashlib
import json

from training.common import DATA, KINDS


def main():
    root=DATA/'sources'/'polyhedral-types'
    if not (root/'audit.json').exists():raise SystemExit('Complete training.acquire_kaggle polyhedral-types first')
    seen={};duplicates=0
    for file in sorted((root/'files').rglob('*')):
        if file.suffix.lower() not in ('.jpg','.jpeg','.png') or file.parent.name not in KINDS:continue
        with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest in seen:
            if seen[digest]['type']!=file.parent.name:raise ValueError('Identical image has conflicting type labels')
            duplicates+=1;continue
        # This set provides no face values; filename counters are not labels.
        seen[digest]=dict(image=str(file.resolve()),type=file.parent.name,value=None,value_trusted=False,
            source='ucffool/dice-d4-d6-d8-d10-d12-d20-images',sha256=digest,
            source_split='train' if 'train' in file.parts else 'valid' if 'valid' in file.parts else 'unknown',
            license='CC0',training_imported=False)
    (root/'type-only.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in seen.values()),encoding='utf-8')
    report=dict(unique_images=len(seen),exact_duplicates=duplicates,
                by_type=dict(Counter(r['type'] for r in seen.values())),face_labels=0,
                imported_to_face_training=False)
    (root/'type-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
