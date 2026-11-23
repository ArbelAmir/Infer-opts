"""Shared helpers: machine-wide benchmark lock, idle check, provenance."""
import contextlib, fcntl, hashlib, json, os, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))
ENGINES = os.environ.get("ENGINES_DIR", os.path.join(ROOT, "engines"))
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
LOCK_PATH = os.environ.get("BENCH_LOCK", "/tmp/infer-opts-bench.lock")


@contextlib.contextmanager
def bench_lock(tag=""):
    """Exclusive machine-wide lock. Everything that burns CPU (benchmarks, builds,
    quantization, evals) must hold it, so two measurements can never overlap."""
    if os.environ.get("BENCH_LOCK_HELD") == "1":  # re-entrant for child processes
        yield
        return
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    t0 = time.time()
    fcntl.flock(fd, fcntl.LOCK_EX)
    waited = time.time() - t0
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()} {tag} {time.strftime('%F %T')}\n".encode())
    os.environ["BENCH_LOCK_HELD"] = "1"
    try:
        yield waited
    finally:
        os.environ.pop("BENCH_LOCK_HELD", None)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _cpu_times():
    with open("/proc/stat") as f:
        p = f.readline().split()[1:]
    v = list(map(int, p))
    return sum(v), v[3] + v[4]


def cpu_busy_fraction(interval=1.0):
    t0, i0 = _cpu_times()
    time.sleep(interval)
    t1, i1 = _cpu_times()
    return 1.0 - (i1 - i0) / max(1, t1 - t0)


def wait_idle(threshold=0.05, timeout=120):
    """Block until the machine is idle (<threshold busy over 1 s). Returns last busy fraction."""
    t0 = time.time()
    while True:
        b = cpu_busy_fraction(1.0)
        if b < threshold or time.time() - t0 > timeout:
            return b
        time.sleep(1)


def sha256_of(path):
    side = path + ".sha256"
    if os.path.exists(side):
        return open(side).read().strip()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    with open(side, "w") as f:
        f.write(h.hexdigest() + "\n")
    return h.hexdigest()


def model_path(name):
    return name if os.path.isabs(name) else os.path.join(MODELS, name)


def build_info(build_dir):
    p = os.path.join(build_dir, "BUILD_INFO.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"build_dir": build_dir, "note": "no BUILD_INFO.json"}


def host_info():
    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    return {
        "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"),
        "ncpu": os.cpu_count(),
        "mem_total_kb": int(sh("awk '/MemTotal/{print $2}' /proc/meminfo") or 0),
        "kernel": sh("uname -r"),
    }


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
