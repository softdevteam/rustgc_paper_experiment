from build import Aggregation, Metric

GCVS_APPENDIX_TABLES = [
    Metric.WALLCLOCK,
    Metric.USER,
]

GCVS_APPENDIX_PLOTS = [
    (Metric.WALLCLOCK, (0.5, 2)),
    (Metric.USER, (0.5, 2)),
]

GCVS_MINI_PLOTS = [
    (Metric.MEM_HSIZE_AVG, (0.5, 6.5)),
    (Metric.WALLCLOCK, (0.6, 1.2)),
]

HEAPSIZES_MINI_PLOTS = [
    # (Metric.MEM_HSIZE_AVG, (0.5, 6.5)),
    (Metric.WALLCLOCK, (0.6, 1.2)),
]

PREMOPT_MINI_PLOTS = [
    (Metric.MEM_HSIZE_AVG, (0.6, 1.4)),
    (Metric.WALLCLOCK, (0.6, 1.4)),
]

ELISION_MINI_PLOTS = [
    (Metric.WALLCLOCK, (0, 1.5)),
    (Metric.USER, (0, 1.5)),
    (Metric.MEM_HSIZE_AVG, (0, 1.5)),
    (Metric.GC_TIME, (0, 1.5)),
    (Metric.TOTAL_COLLECTIONS, (0, 1.5)),
]

ELISION_APPENDIX_PLOTS = [
    (Metric.WALLCLOCK, (0, 1.2)),
    (Metric.USER, (0, 1.2)),
    (Metric.GC_TIME, (0, 1.2)),
    (Metric.MEM_HSIZE_AVG, (0.5, 1.6)),
    (Metric.TOTAL_COLLECTIONS, (0.1, 3)),
    (Metric.OBJ_ALLOCD_GC, (0.8, 1.1)),
]

ELISION_SUMMARY_TABLES = [
    (Metric.PCT_ELIDED, Aggregation.SUITE_ARITH),
]

ELISION_APPENDIX_TABLES = [
    Metric.PCT_ELIDED,
    Metric.WALLCLOCK,
    Metric.USER,
    Metric.TOTAL_COLLECTIONS,
    Metric.MEM_HSIZE_AVG,
]
