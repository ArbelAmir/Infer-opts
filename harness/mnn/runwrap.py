#!/usr/bin/env python3
"""runwrap.py <log> <cmd...>: run cmd, tee stdout+stderr to log, print 'WRAP wall_s= maxrss_kb= rc='."""
import sys, subprocess, resource, time
log = sys.argv[1]; cmd = sys.argv[2:]
t = time.time()
with open(log, "w") as f:
    rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
w = time.time() - t
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
with open(log, "a") as f:
    f.write(f"\nWRAP wall_s={w:.3f} maxrss_kb={rss} rc={rc}\n")
print(f"WRAP wall_s={w:.3f} maxrss_kb={rss} rc={rc}")
