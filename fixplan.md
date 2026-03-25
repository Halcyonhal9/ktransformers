# Fix: TaskQueue worker thread 100% CPU spin when idle

## Context

GitHub issue [#1858](https://github.com/kvcache-ai/ktransformers/issues/1858): `TaskQueue::worker()` busy-waits in a tight loop when no tasks are queued, pinning one CPU core at 100% even when the server is fully idle. `sync()` has the same problem — pure spin on an atomic load. The header already includes `<condition_variable>` and `<mutex>` but they are unused.

## Files to modify

1. `kt-kernel/cpu_backend/task_queue.h` — add `std::mutex` and `std::condition_variable` members
2. `kt-kernel/cpu_backend/task_queue.cpp` — use CV in `worker()`, `enqueue()`, `sync()`, and `~TaskQueue()`

## Race condition analysis (why the naive fix is wrong)

A naive approach — just adding `cv.wait()` in the worker and `cv.notify_one()` in enqueue without mutex coordination — has a **lost-wakeup race**:

1. Worker checks predicate → false (queue empty), still holding mutex
2. Enqueuer stores new node via atomic (no mutex needed)
3. Enqueuer calls `cv.notify_one()` — **worker hasn't entered `wait()` yet, notification lost**
4. Worker enters `cv.wait()` — blocks forever, task never processed

The same race exists for `sync()`: the worker decrements `pending` and notifies, but if `sync()` is between its predicate check and `wait()`, the notification is lost.

**Fix:** After each state change, acquire-then-release the mutex before notifying. This serializes with the waiter's predicate-check-to-wait transition, guaranteeing the notification is either unnecessary (predicate already true) or delivered (waiter is in `wait()`).

## Changes

### `task_queue.h` — add two private members

```cpp
std::mutex mtx;
std::condition_variable cv;
```

### `task_queue.cpp` — full corrected implementation

**`enqueue()`** — lock-unlock the mutex after the atomic store, then notify:

```cpp
void TaskQueue::enqueue(std::function<void()> task) {
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(task);
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);  // serialization point
  }
  cv.notify_one();  // wake idle worker
}
```

Why: if the worker is between predicate-check and `wait()` (holding the mutex), the enqueuer blocks on `lock(mtx)` until the worker enters `wait()` (which atomically releases the mutex). Then `notify_one()` delivers.

**`worker()`** — CV-wait when queue is empty; hold mutex around `pending` decrement:

```cpp
void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      if (next->task) {
        next->task();
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx);
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
      cv.notify_all();  // wake sync() waiters
    } else {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
  }
}
```

Why mutex around `pending.fetch_sub`: prevents a lost-wakeup race with `sync()`. If `sync()` checks `pending > 0` and is about to `wait()`, the worker blocks on the mutex until `sync()` enters `wait()`, then decrements and notifies.

**`~TaskQueue()`** — notify worker so it can observe `done` and exit:

```cpp
TaskQueue::~TaskQueue() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    done.store(true, std::memory_order_release);
  }
  cv.notify_all();
  if (workerThread.joinable()) workerThread.join();

  Node* node = head.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}
```

**`sync()`** — CV-wait instead of spin:

```cpp
void TaskQueue::sync(size_t allow_n_pending) {
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [&] {
    return pending.load(std::memory_order_acquire) <= allow_n_pending;
  });
}
```

## Performance impact

The mutex lock/unlock adds ~50ns per task enqueue and per task completion. Tasks are MoE expert forward passes taking tens of milliseconds each, so this overhead is unmeasurable. The lock-free linked-list structure is preserved — the mutex only guards the CV sleep/wake transitions.

## Verification

### Automated tests (existing CI)

The repo has a CI pipeline (`.github/workflows/kt-kernel-tests.yml`) that runs:
```bash
cd kt-kernel && bash install.sh build
cd kt-kernel/test && python3 run_suite.py --hw cpu --suite default
```

Tests live in `kt-kernel/test/per_commit/test_basic_cpu.py`. These test CPUInfer initialization and basic module functionality. They exercise the TaskQueue indirectly (CPUInfer creates a TaskQueue in its constructor). Run these to confirm nothing is broken:

```bash
cd kt-kernel && bash install.sh build
cd kt-kernel/test && python3 run_suite.py --hw cpu --suite default
```

There are also integration examples that exercise enqueue + sync end-to-end:
```bash
cd kt-kernel/examples
python3 test_linear.py    # calls CPUInfer.sync()
python3 test_attention.py # calls CPUInfer.sync()
```

These require a built kt-kernel and appropriate hardware (CPU with AMX support for some).

### Manual verification on your AI server

**1. Build and install the patched kt-kernel:**
```bash
cd kt-kernel && bash install.sh build
# or: pip install -e .
```

**2. Launch your sglang server as usual** with `--kt-cpuinfer` flags.

**3. Idle CPU check** — the primary test. With no requests in flight:
```bash
# Check per-thread CPU usage
pidstat -t -p $(pgrep -f sglang) 1 5

# Or use top in thread mode
top -H -p $(pgrep -f sglang)
```

**Before fix:** one `sglang::schedul` thread at ~100% CPU.
**After fix:** all threads at ~0% CPU when idle.

**4. Functional smoke test** — confirm inference still works:
```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

Verify you get a valid response.

**5. Latency regression check** (optional but recommended):
```bash
# Run a few requests and note time-to-first-token
# The CV wakeup adds ~5-15μs, which should be undetectable
# compared to the tens-of-ms MoE forward passes
```

**6. Post-request idle check** — after completing requests, verify the thread drops back to ~0% CPU:
```bash
# Send a request, wait for completion, then check again
pidstat -t -p $(pgrep -f sglang) 1 10
```

**7. Shutdown test** — verify the server shuts down cleanly (no hangs from the CV):
```bash
kill -SIGTERM $(pgrep -f sglang)
# Should exit within a few seconds
```
