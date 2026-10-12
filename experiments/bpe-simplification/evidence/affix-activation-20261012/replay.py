from pathlib import Path
import argparse,subprocess,shutil,os
parser=argparse.ArgumentParser(description='Build a frozen normal or allocation diagnostic arm in a disposable checkout.')
parser.add_argument('--checkout',type=Path,required=True)
parser.add_argument('--target-dir',type=Path,required=True)
parser.add_argument('--arm',choices=['baseline', 'flags', 'candidate-choices', 'candidate-paired', 'candidate-cursor', 'candidate-chars'],required=True)
parser.add_argument('--allocation-diagnostic',action='store_true')
parser.add_argument('--offline',action='store_true')
args=parser.parse_args();repo=args.checkout.resolve();target=args.target_dir.resolve();e=Path(__file__).resolve().parent
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert head=='a2b20f329d1b4a0d39b5721f4a7a678a80d0d9b2','Use a detached disposable checkout at report base SHA'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),'Checkout must be clean before applying the frozen source'
assert repo!=e and repo not in e.parents,'Use a separate checkout'
source=e/('diagnostic-source' if args.allocation_diagnostic else 'source')/args.arm
assert source.is_dir(),source
shutil.copytree(source,repo,dirs_exist_ok=True)
command=['cargo','build','--release','--locked','--manifest-path',str(repo/'experiments/bpe-simplification/runner/Cargo.toml'),'--target-dir',str(target)]
if args.offline:command.append('--offline')
subprocess.run(command,cwd=repo,env=os.environ|{'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'2'},check=True)
print(target/'release/bpe-bench-runner')
