from invoke import Collection, task

from bench import bench, clean_results
from build import build_benchmarks, clone

ns = Collection()
ns.add_task(clone)
ns.add_task(build_benchmarks)

# Bench
ns.add_task(bench)
ns.add_task(clean_results)


@task
def all(c):
    print("Hello")


ns.add_task(all)
