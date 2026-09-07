# Radiance attention softmax

One lane processes an attention row. The kernel scales scores, applies causal
and sliding-window masks, subtracts the row maximum, interpolates an exponential
lookup table, and normalizes the valid entries. Masked entries are exactly zero.
The original float32 algorithm and default shape (one head, sequence length 2)
are preserved.

## Fixes

- Use the repository's `../../radiance-support/common.mk`; allow `PYTHON_REF`
  to be supplied by the caller. Resolve the optional Gemmini include from
  `RADIANCE_LIB_PATH` so an absent local runtime cannot produce a bare `-I`.
- Restore correctness failure reporting through per-lane status followed by
  a check in the manager lane after `mu_schedule`. Putting the infinite failure
  loop inside the callback caused this LLVM compiler to omit the divergent
  row-bound guard: inactive rows overwrote adjacent data. Keeping the callback
  returning normally allows correct split/join generation.
- Treat NaN differences as failures as well as excessive finite differences.
- Parameterize the reference generator and export true PyTorch results in
  `reference.json`. The kernel's existing LUT-reference tolerance is 1e-3;
  independent RTL verification uses the tighter 2e-6 absolute tolerance from
  the Python LUT accuracy check.

## Reproduce on this machine

```
/home/prashanth/chipyard/.vecadd-sim/run-slice-softmax.sh
```

The script stages sources into the writable Chipyard worktree, builds with the
Muon LLVM compiler and single-core runtime, and runs Verilator
`MuonCoreLongTestConfig`. It uses the simulator/runtime fixes established during
the earlier vecadd/softmax work. No additional RTL changes were needed here.
This is a one-core, 16-lane, one-warp testbench run, not a full Chipyard SoC run.
The script expects the prebuilt simulator, runtime, and PyTorch environment.

Results, ELF files, inputs, reference outputs, build logs, simulation logs and
SQLite RTL memory traces are saved under
`/home/prashanth/chipyard/.vecadd-sim/slice-softmax/results/`.

| Case | Heads | Sequence | Window | Outputs |
|---|---:|---:|---:|---:|
| default | 1 | 2 | 512 | 4 |
| causal14 | 1 | 14 | 512 | 196 |
| sliding | 2 | 33 | 8 | 2178 |

All three completed with simulator exit code 0 and passed independent comparison.
Maximum absolute error against PyTorch was 4.76837158203125e-7 in each case.
The verifier reconstructs final output bytes from RTL stores, checks complete
coverage and finite values, and checks masks and row sums. It can be run separately:

```
python3 verify_trace.py softmax.sqlite kernel.radiance.elf reference.json --nm /path/to/llvm-nm
```

A negative control with the first expected value changed from 1 to 0 failed
with the intended simulator timeout after 10,001 cycles (exit 134), confirming
that correctness failures prevent normal completion. Its artifacts are in
`/home/prashanth/chipyard/.vecadd-sim/slice-softmax/negative/`.
