from classify_runs_fit import classify_runs
from changepoint_analysis_fit import changepoint_analysis
import time

if __name__ == '__main__':
    t1 = time.time()
    changepoint_analysis(S=1.0, curve="convex", direction="decreasing")
    t2 = time.time()
    classify_runs(es=0.05, significance=0.05)
    t3 = time.time()

    print(f'changepoint analysis: {t2-t1}s')
    print(f'classify runs: {t3-t2}s')
    print(f'total runtime: {t3-t1}s')
