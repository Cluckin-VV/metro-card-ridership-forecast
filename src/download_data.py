"""Fetch the pinned public Kaggle v45 snapshot and verify it before saving.

Existing raw files are never overwritten. No account credentials are needed.
"""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import io
import hashlib

ROOT=Path(__file__).resolve().parents[1]
URL='https://www.kaggle.com/api/v1/datasets/download/maheshshantaram/bengaluru-nammametro-daily-ridership?datasetVersionNumber=45'
CSV='NammaMetro_Ridership_Dataset.csv'
EXPECTED='21a6efb3b61e45fff936ccc57e825b900e357107dec886d0202fef97aea49316'

def main():
    folder=ROOT/'data/raw';folder.mkdir(parents=True,exist_ok=True)
    target=folder/CSV
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest()==EXPECTED, 'Existing CSV differs from pinned snapshot'
        print('Verified existing v45 CSV. Download not required.');return
    request=Request(URL,headers={'User-Agent':'NCUT-course-experiment/1.0'})
    with urlopen(request,timeout=90) as response:payload=response.read()
    with ZipFile(io.BytesIO(payload)) as archive:
        matches=[n for n in archive.namelist() if Path(n).name==CSV]
        if len(matches)!=1:raise ValueError('Expected CSV missing or ambiguous in downloaded archive')
        raw=archive.read(matches[0])
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('Snapshot hash mismatch; do not use unverified replacement')
    archive_path=folder/'kaggle_v45.zip'
    if not archive_path.exists():archive_path.write_bytes(payload)
    with target.open('xb') as out:out.write(raw)
    print('Downloaded and verified Kaggle v45.')

if __name__=='__main__':main()
