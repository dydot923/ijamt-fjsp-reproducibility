"""Verify the original distributed code tree without scientific dependencies.

FILE_MANIFEST.json describes version0.1.0 before publication. Legitimate edits
after publication require a new reviewed manifest for a new release.
"""
from pathlib import Path
import ast, hashlib, json, re

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'FILE_MANIFEST.json').read_text('utf8'))
    for row in manifest:
        path=(ROOT/row['file']).resolve()
        assert path.is_relative_to(ROOT.resolve()),row['file']
        data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['sha256'],row['file']
        assert len(data)==row['bytes'],row['file']
        assert len(data)<25*1024*1024,row['file']
        assert path.suffix.lower() not in ['.docx','.doc','.zip','.7z','.pyc'],row['file']
        if path.suffix=='.py':ast.parse(data.decode('utf8'),filename=row['file'])
        if path.suffix in ['.py','.json','.md','.txt','.cff']:
            s=data.decode('utf-8-sig')
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16})',s),row['file']
            assert not re.search(r'[CDE]:[/\\](?:Users|Dydot)',s,re.I),row['file']
    assert not (ROOT/'replication/data/v42_dataset.json').exists(),'Additional data are outside this code-only manifest'
    print(json.dumps({'status':'PASS','manifest_files':len(manifest),'examples':6,
                      'public_input_scope':'constructed validation only','license_selected':(ROOT/'LICENSE').exists()}))


if __name__=='__main__':main()
