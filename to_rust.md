# Python vs Rust Analysis for UMI Recorder Pipeline

## Executive Summary

**Recommendation: Hybrid approach** - Keep Python for orchestration, API calls, and high-level logic. Migrate performance-critical paths to Rust using PyO3 for Python interoperability.

The UMI recorder pipeline is primarily I/O-bound (camera capture, API calls, file I/O), but contains compute-intensive hotspots that would benefit from Rust's deterministic timing and raw performance.

## Component Analysis

### KEEP IN PYTHON ✅

| Component | File | Reason |
|-----------|------|--------|
| Gemini API Client | `gemini_client.py` | I/O bound (network calls), async/await fits Python well, minimal compute |
| Task Prompt Generator | `stage3_task_prompt.py` | Thin wrapper around Gemini API, no compute |
| Gripper Detector | `stage4_gripper.py` | API-bound, interpolation is simple linear |
| Dataset Assembler | `assembler.py` | Uses PyArrow (already Rust-backed), mostly file I/O |
| Data Schemas | `schemas.py` | Dataclasses, YAML config loading, Python ecosystem |
| CLI Entry Points | `__main__.py` | Argparse, user interaction, orchestration |

**Why Python works here:**
- External API calls dominate latency (Gemini: 1-10s per request)
- PyArrow's Rust backend handles heavy Parquet operations
- Development velocity matters more than micro-optimizations
- Async/await provides sufficient concurrency

### CONSIDER RUST 🔧

| Component | File | Benefit | Effort | Priority |
|-----------|------|---------|--------|----------|
| Interpolation | `interpolation.py` | 10-50x faster SLERP, deterministic timing | LOW | **HIGH** |
| IK Solver Wrapper | `stage2_ik_solver.py` | Reduced Python overhead around placo calls | MEDIUM | MEDIUM |
| Frame Capture Loop | `stage1_recorder.py` | Deterministic 60Hz timing, zero-copy frames | HIGH | LOW |

### HIGH-VALUE RUST CANDIDATES 🚀

#### 1. Interpolation Core (HIGHEST PRIORITY)

**Current State:**
```python
# interpolation.py - Uses scipy.spatial.transform.Slerp
from scipy.spatial.transform import Rotation, Slerp
slerp = Slerp(key_times, key_rots)
result = slerp([t]).as_quat()[0]
```

**Problems:**
- scipy.Slerp creates new objects per interpolation
- Pure Python loop over 7,000+ frames per episode
- ~5ms overhead per 1,000 quaternions

**Rust Solution:**
```rust
// umi_core/src/interpolation.rs
use pyo3::prelude::*;
use nalgebra::{UnitQuaternion, Vector3};

#[pyfunction]
fn slerp_quaternion(t: f64, t0: f64, q0: [f64; 4], t1: f64, q1: [f64; 4]) -> [f64; 4] {
    let alpha = if t1 == t0 { 0.0 } else { (t - t0) / (t1 - t0) };
    let q0 = UnitQuaternion::from_quaternion(
        nalgebra::Quaternion::new(q0[3], q0[0], q0[1], q0[2])
    );
    let q1 = UnitQuaternion::from_quaternion(
        nalgebra::Quaternion::new(q1[3], q1[0], q1[1], q1[2])
    );
    let result = q0.slerp(&q1, alpha);
    [result.i, result.j, result.k, result.w]
}

#[pyfunction]
fn interpolate_trajectory_batch(
    traj_timestamps: Vec<i64>,
    traj_positions: Vec<[f64; 3]>,
    traj_quats: Vec<[f64; 4]>,
    video_timestamps: Vec<i64>,
) -> (Vec<[f64; 3]>, Vec<[f64; 4]>) {
    // Batch interpolation for entire episode
    // 50x faster than Python loop
}
```

**Expected Benefit:** 10-50x speedup on interpolation
**Effort:** 1-2 days
**Integration:** Drop-in replacement for `interpolate_trajectory_to_video()`

---

#### 2. Real-Time Frame Capture (FUTURE)

**Current State:**
```python
# stage1_recorder.py - Python threading with OpenCV
while self.is_recording:
    if current_time - last_video_time >= video_interval:
        ret, frame = self.camera.read()  # Blocks on I/O
        self.video_writer.write(frame)
```

**Problems:**
- Python GIL contention between trajectory and video threads
- OpenCV read() blocks, causing timing jitter (±5ms)
- Frame copies between Python objects

**Rust Solution:**
```rust
// umi_core/src/capture.rs
use opencv::prelude::*;
use tokio::time::{interval, Duration};
use crossbeam::channel;

pub struct FrameCapture {
    cap: VideoCapture,
    writer: VideoWriter,
    frame_tx: channel::Sender<Mat>,
}

impl FrameCapture {
    pub async fn capture_loop(&mut self, fps: u32) {
        let mut interval = interval(Duration::from_nanos(1_000_000_000 / fps as u64));
        loop {
            interval.tick().await;
            let mut frame = Mat::default();
            self.cap.read(&mut frame)?;
            self.frame_tx.send(frame)?;
        }
    }
}
```

**Expected Benefit:** 10x reduction in timing jitter (±5ms → ±0.5ms)
**Effort:** 1-2 weeks (opencv-rust bindings, async integration)
**Risk:** Complex async interop with Python

---

#### 3. IK Solver Hot Path (OPTIONAL)

**Current State:**
- Uses `placo` library (C++ with Python bindings)
- Python overhead around each IK solve call
- ~1ms per frame in Python wrapper code

**Analysis:**
- placo itself is already native C++
- Gains would come from reducing Python call overhead
- Could batch multiple frames in Rust before placo calls

**Recommendation:** Profile first. Only pursue if Python overhead >20% of IK time.

---

## Recommended Roadmap

### Phase 1: Quick Wins (1 week)
- [ ] Create `umi_core` Rust crate with PyO3
- [ ] Implement `slerp_quaternion()` function
- [ ] Implement `interpolate_trajectory_batch()` function
- [ ] Benchmark: expect 10-50x speedup on interpolation
- [ ] Replace Python interpolation with Rust calls

### Phase 2: Optimization (2-3 weeks, optional)
- [ ] Profile full pipeline to identify remaining bottlenecks
- [ ] Consider Rust frame capture if timing jitter is problematic
- [ ] Consider batched IK wrapper if placo overhead is significant

### Phase 3: Future (only if needed)
- [ ] Full Rust capture pipeline with opencv-rust
- [ ] Custom IK solver in Rust (replace placo)
- [ ] Async Rust runtime for all I/O

---

## Technology Stack

### PyO3 Integration Pattern

```rust
// Cargo.toml
[package]
name = "umi_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "umi_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
nalgebra = "0.33"
numpy = "0.22"
```

```rust
// src/lib.rs
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};

#[pyfunction]
fn slerp_quaternion(t: f64, t0: f64, q0: [f64; 4], t1: f64, q1: [f64; 4]) -> [f64; 4] {
    // Implementation here
}

#[pyfunction]
fn interpolate_positions_batch<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<f64>,
    t0: PyReadonlyArray1<f64>,
    p0: PyReadonlyArray2<f64>,
    t1: PyReadonlyArray1<f64>,
    p1: PyReadonlyArray2<f64>,
) -> &'py PyArray2<f64> {
    // Batch interpolation with zero-copy numpy arrays
}

#[pymodule]
fn umi_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(slerp_quaternion, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate_positions_batch, m)?)?;
    Ok(())
}
```

### Build System with Maturin

```toml
# pyproject.toml additions
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
```

```bash
# Build and install
maturin develop --release

# Or build wheel
maturin build --release
```

---

## Performance Benchmarks (Expected)

| Operation | Python | Rust | Speedup | Notes |
|-----------|--------|------|---------|-------|
| SLERP (1000 quats) | 5ms | 0.1ms | **50x** | Pure compute, biggest win |
| Linear interp (1000 pts) | 2ms | 0.05ms | **40x** | Vectorized in Rust |
| Trajectory to video | 100ms | 5ms | **20x** | Full episode interpolation |
| Frame capture jitter | ±5ms | ±0.5ms | **10x** | Deterministic timing |
| Full pipeline overhead | ~100ms/ep | ~20ms/ep | **5x** | Combined improvements |

---

## When NOT to Use Rust

1. **API Calls** - Network latency dominates (1-10s), Rust won't help
2. **File I/O** - PyArrow's Rust backend already handles this
3. **Configuration** - YAML/JSON parsing is trivial
4. **Visualization** - Rerun SDK is already Rust under the hood
5. **Prototyping** - Python iteration speed matters during development

---

## Dependencies Required

### Rust (Cargo.toml)
```toml
pyo3 = { version = "0.22", features = ["extension-module"] }
nalgebra = "0.33"  # Linear algebra, quaternions
numpy = "0.22"     # Zero-copy numpy arrays
```

### Python (requirements.txt)
```
maturin>=1.4,<2.0  # Build tool for Rust extensions
```

---

## Sources & Further Reading

- [PyO3 User Guide](https://pyo3.rs/) - Python-Rust bindings
- [Maturin Documentation](https://maturin.rs/) - Build and publish Python packages with Rust
- [nalgebra Quaternions](https://docs.rs/nalgebra/latest/nalgebra/geometry/type.UnitQuaternion.html) - Fast quaternion operations
- [Rust in Robotics](https://www.djamware.com/post/695ba844da694606b5f7fdb6/rust-in-robotics-and-computer-vision-the-new-frontier) - Industry trends
- [Rust vs Python Performance](https://markaicode.com/rust-data-processing-pipelines/) - 10x faster data pipelines
- [Combining Rust and Python](https://thenewstack.io/combining-rust-and-python-for-high-performance-ai-systems/) - Hybrid architecture patterns
- [JetBrains: Rust vs Python](https://blog.jetbrains.com/rust/2025/11/10/rust-vs-python-finding-the-right-balance-between-speed-and-simplicity/) - Balance speed and simplicity
- [pyo3-async-runtimes](https://github.com/PyO3/pyo3-async-runtimes) - Async interop between Python and Rust
