#!/usr/bin/env python3
"""Build the pinned PR with serial feed and an explicit four-worker merge pool."""
import argparse
from pathlib import Path
import subprocess
from build_profiled import PR_HEAD, build

parser = argparse.ArgumentParser()
parser.add_argument('pr_checkout', type=Path)
args = parser.parse_args()
source = args.pr_checkout.resolve()
head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
if head != PR_HEAD or dirty:
    raise SystemExit('PR checkout must be clean at the pinned head')
print(build('pr4',source,False,training_workers=4))
