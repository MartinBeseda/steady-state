from classify_runs_fit import classify_runs
from changepoint_analysis_fit import changepoint_analysis
import time


def run_cpsssd_with_params(S1, curve, direction, es, significance, interp_method, poly_degree, changepoints_dir,
                           classification_dir):
    t1 = time.time()
    changepoint_analysis(S=S1, curve=curve, direction=direction, interp_method=interp_method, poly_degree=poly_degree,
                         changepoints_path=changepoints_dir)
    t2 = time.time()
    classify_runs(es=es, significance=significance, changepoints_dir=changepoints_dir,
                  classification_dir=classification_dir)
    t3 = time.time()

    print(f'changepoint analysis: {t2-t1}s')
    print(f'classify runs: {t3-t2}s')
    print(f'total runtime: {t3-t1}s')


if __name__ == '__main__':
    # run_cpsssd_with_params(S1=1.0, curve='convex', direction='decreasing', es=0.05, significance=0.05)
    pass
