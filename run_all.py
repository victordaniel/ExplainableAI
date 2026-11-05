import time
from joblib import Parallel, delayed
from audit.run import run
tasks = [(ds, r) for r in range(10) for ds in ('adult', 'acs_income', 'compas', 'german')]
t0 = time.time()
for out in Parallel(n_jobs=4, return_as='generator_unordered')(delayed(run)(ds, r) for ds, r in tasks):
    print(out, 'elapsed %.0f s' % (time.time() - t0), flush=True)
print('ALL DONE', flush=True)
