from run_selected_full_helper import measure, busy
from run_pretokenizers import prepare, inputs

assert not busy(), busy()
prepare()
for info in inputs():
    for arm in ('main', 'baseline', 'candidate'):
        measure(info, arm, 'selected-phases', 1)
for info in inputs():
    for arm in ('main', 'baseline', 'candidate'):
        measure(info, arm, 'phases-pipeline', 1)
print('Selected pretokenizer phase contrast complete', flush=True)
