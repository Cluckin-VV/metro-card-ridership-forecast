"""Re-run deterministic core experiment once and independently check delivery data."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'qa';QA.mkdir(exist_ok=True)
FILES=['predictions.csv','regression_cv.csv','classification_cv.csv','clusters.csv','next_forecast.json','data_audit.json','split_manifest.json']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    before={n:digest(ROOT/'results'/n) for n in FILES}
    baseline=QA/'reproduction_baseline';baseline.mkdir(exist_ok=True)
    for n in FILES:(baseline/n).write_bytes((ROOT/'results'/n).read_bytes())
    proc=subprocess.run([sys.executable,str(ROOT/'src/experiment.py')],cwd=ROOT,capture_output=True,encoding='utf-8',errors='replace')
    (QA/'reproduction.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8');proc.check_returncode()
    after={n:digest(ROOT/'results'/n) for n in FILES}
    # Parallel tree reductions can vary at floating-point round-off level.
    # Require identical schemas, rows and labels, with explicit numeric tolerance.
    for n in FILES:
        if n.endswith('.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(baseline/n),pd.read_csv(ROOT/'results'/n),check_exact=False,rtol=1e-12,atol=1e-8)
        else:
            assert json.loads((baseline/n).read_text())==json.loads((ROOT/'results'/n).read_text()),n
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,encoding='utf-8',errors='replace')
    (QA/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8');tests.check_returncode()
    df=pd.read_csv(ROOT/'results/predictions.csv')
    test=df.loc[df.split=='test']
    independent_mae=float((test.actual-test.predicted).abs().mean())
    metrics=json.loads((ROOT/'results/metrics.json').read_text())
    assert abs(independent_mae-metrics['test']['regression']['MAE'])<1e-8
    model=pd.read_csv(ROOT/'data/processed/model_table.csv')
    assert model.split.value_counts().to_dict()=={'development':303,'test':75,'calibration':59,'embargo':1}
    report={'result_values_reproduced':True,'numeric_tolerance':{'rtol':1e-12,'atol':1e-8},
            'byte_identical_files':[n for n in FILES if before[n]==after[n]],'sha256':after,'tests_exit_code':tests.returncode,
            'test_MAE_recomputed':independent_mae,'max_test_data_age_days':float(test.data_age_days.max()),
            'calibration_samples':59,'embargo_date':'2026-04-30','test_samples':75,
            'original_raw_sha256':digest(ROOT/'data/raw/NammaMetro_Ridership_Dataset.csv')}
    (QA/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
