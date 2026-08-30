# Eternity II — GPU port of Blackwood's solver

A CUDA-native port of [Joshua Blackwood's Eternity II solver](https://github.com/jblackwood345/EternityII_Solver),
running his chronological-backtracking search as thousands of independent depth-first searches
on the GPU, with each thread's search state persisting across kernel launches.

The [Eternity II puzzle](https://en.wikipedia.org/wiki/Eternity_II_puzzle) is a 16×16
edge-matching puzzle with 480 internal edges. It has never been solved. The $2,000,000 prize
went unclaimed when the competition closed in 2010, and no complete solution has been published
since.

## Results

| | |
|---|---|
| Blackwood's own record (CPU) | 470/480 (10 conflicts) |
| A perfect solution | 480/480 (0 conflicts) |

The effort to reach a solution is not merely a matter of more compute — see
[eternity2.dev on why a faster computer doesn't help](https://eternity2.dev/research/why/prune-vs-speed/).

## Requirements

- NVIDIA GPU with CUDA support, and the CUDA toolkit (`nvcc`) on `PATH`
- JDK 21+
- Maven
- MSVC toolchain on Windows (`nvcc` needs `cl.exe` even for PTX-only output)

## Build and run

```bash
# 1. Compile the CUDA kernel to PTX (once, and after any change to the .cu).
#    Set -Arch to your GPU's compute capability if compute_120 is rejected.
powershell -File build-kernel.ps1 -Arch compute_120

# 2. Build the Java side.
mvn clean package

# 3. Run from the repository root (pieces.csv is read by relative path).
mvn exec:java -Dexec.mainClass=dk.puzzle.blackwood.BlackwoodGpuRunner
```

or, once built, directly:

```bash
java -cp "target/classes;$(cat cp.txt)" dk.puzzle.blackwood.BlackwoodGpuRunner
```

Saved boards land in `~/EternitySolutions_GPU/` as three files per board: a bucas-linked
raw board, a physical piece layout, and a Blackwood-numbered baseboard usable as a future seed.

## Configuration

All optional, all environment variables:

| Variable | Default | Effect |
|---|---|---|
| `ETERNITY_GPU_NUM_THREADS` | `1024` | Independent DFS searches. **16384 measured best** at production scale; the default is conservative. |
| `ETERNITY_GPU_SOLUTIONS_DIR` | `~/EternitySolutions_GPU` | Where boards are saved. |
| `ETERNITY_GPU_SEEDING` | **disabled** | `true` resumes attempts from saved boards instead of starting every one from a random corner (see below). |
| `ETERNITY_GPU_SEED_SAMPLING` | enabled | `false` picks the seed pool as a strict top-K by depth. Enabled, it draws the pool by depth-weighted random sampling, so restarts don't all resume from the identical elite boards (see below). |
| `ETERNITY_GPU_FRESH_FRACTION` | `40` | Percent of attempts that ignore seeds and start from a random corner. The explore/exploit dial (see below). |
| `ETERNITY_GPU_MAX_RETREAT` | `100` | How far back from a seed's tip a resuming thread may pull. Drawn uniformly in `[0, N]`, so the mean pull-back is `N/2` (see below). |
| `ETERNITY_GPU_SHARED_CACHE` | enabled | `false` reads the four hot per-step tables from `__constant__` rather than a `__shared__` copy. |
| `ETERNITY_NODE_CAP` | `50000000000` | Nodes before a CPU-side attempt restarts. Blackwood's own value; 25B/50B/100B measured indistinguishable. |
| `ETERNITY_DRIVE_UPLOAD` | enabled | `false` disables Google Drive mirroring (see below). |

## Thread count

The single most consequential setting. Measured head-to-head at production scale, equal
wall-clock, scoring real post-completion conflict counts rather than raw depth:

| Threads | Best result |
|---|---|
| 64 | never below 15 conflicts, over 22.7 h |
| 1024 | 13 conflicts, over 30.4 h |
| **16384** | **12 conflicts, in ~7 h** |

This reversed an earlier prediction made from short fixed-node-budget benchmarks, which had
suggested high thread counts would underperform. They do lose on *depth per lineage* — but depth
is not the metric that matters, and only the full-scale conflict-scored comparison showed it.

## Google Drive mirroring (optional)

`dk.puzzle.io.drive` mirrors each saved board to Google Drive as a small text record, so a long
unattended run can be checked remotely. It is entirely optional and deliberately easy to remove:

1. **Off:** set `ETERNITY_DRIVE_UPLOAD=false`.
2. **Out:** delete the single `DriveUploader.uploadRecord(...)` call in `BlackwoodGpuRunner`.
3. **Gone:** delete the `dk.puzzle.io.drive` package, that call, and the three `com.google.*`
   dependencies in `pom.xml`.

No credentials are committed. Without them the first upload fails, logs one warning, and disables
itself for the rest of the run — so a fresh clone works out of the box with no Drive setup.

## How it works

Each GPU thread runs one independent instance of Blackwood's search: place pieces in a fixed
board order, drawing from candidate tables keyed by the left and bottom edge colours, subject to
a *break schedule* (how many mismatched edges are permitted by each depth) and a heuristic gate
on rare colours.

Two things make it work on a GPU:

- **Persistent per-thread state.** Thread state is checkpointed to global memory at the end of
  every launch and reloaded at the start of the next, so a search continues across launches
  instead of restarting. Without this, no thread could ever search deeper than a single launch's
  node budget, no matter how long the process ran.
- **One DFS per thread, not per warp.** Candidate fan-out is small enough that dedicating 32
  lanes to a single search node wastes most of them.

Search state resets only at *epoch* boundaries (`EPOCH_LAUNCHES`, default 20,000), when candidate
tables are rebuilt and re-randomised.

### Seeding is off by default

Resuming attempts from saved boards lets the search start at the frontier instead of re-deriving
250 pieces of known progress — but the project's own A/B testing found that, however that resuming
is tuned, it never once produced a board better than what was already in the seed pool. A
[`BlackwoodGpuFreshFractionHarness`](../Eternity/src/test/java/dk/puzzle/blackwood/BlackwoodGpuFreshFractionHarness.java)
run (4 arms x 180s, identical 59-board seed pool) found best-board results tied across 0/25/50%
fresh-start mixing (all 13) with median quality degrading monotonically as more exploration was
mixed in; a separate retreat distance A/B found the same shape — perturbing further from a seed
made results more diverse and *worse*, never better. That is the signature of a narrow local
optimum: reliable at reproducing what is already known, never observed to exceed it. It matches
Blackwood's own account of his 470/480 record — a month of continuous, unseeded random-restart
search on one PC, his own word for it "luck," not refinement of a near-miss.

Over a short horizon seeded search does clearly win (the same harness measured unseeded's best at
18 against seeded's 13, since a cold start can't climb to competitive depth in 180 seconds), so
this is a bet on where the ceiling is, not a claim that seeding is worthless. If the goal is a
steady stream of already-known-quality boards, re-enable it. If the goal is finding something past
what the pool already has, exploiting a peak that has never yielded anything better is the wrong
tool regardless of how well the exploitation is tuned.

Set `ETERNITY_GPU_SEEDING=true` to resume the old behaviour. Three things exist to make seeded runs
less repetitive if you do:

- **Persistent duplicate suppression.** Completed boards are fingerprinted (SHA-256 of the bucas
  encoding) into `.saved_completed_boards` in the output directory, so a board found in an earlier
  run is not saved or uploaded again. Deduping on the *completed* board matters: two different
  partial boards in one lineage can complete to the identical 256-piece result. With this measuring
  it for the first time, the actual repeat rate turned out to be **80%** — over 352 launches, 10
  boards cleared the conflict threshold and 8 were boards already found in earlier runs.
- **Depth-weighted seed sampling** (`ETERNITY_GPU_SEED_SAMPLING`, on by default *within* seeded
  mode). The pool is drawn by weighted random sampling rather than a strict top-K, so each run
  resumes from a different slice instead of the identical elite boards every time.
- **`ETERNITY_GPU_FRESH_FRACTION` and `ETERNITY_GPU_MAX_RETREAT`.** Percent of attempts starting
  fresh, and how far a resuming thread pulls back from its seed's tip (drawn uniformly in `[0, N]`,
  so the mean pull-back is `N/2`). These are exactly the dials the A/B above already swept — raising
  them did not find better boards, only worse medians at an unchanged best. Don't raise either to
  fight the duplicate rate expecting it to surface something new; the evidence above says it won't.

### Break schedule

`BwUtil.BREAK_INDEXES_ALLOWED` is a cumulative per-depth **ceiling**: mismatches are forbidden
below depth 201, and one more becomes permitted at each listed depth. This port ships Blackwood's
original 10-entry schedule:

```
201, 206, 211, 216, 221, 225, 229, 233, 237, 239
```

It is a ceiling, not a quota — nothing obliges the search to spend its allowance, and candidates
are scored `score - 100000 * breakCount`, so break-free placements are always tried first and a
break is taken only when the search is otherwise stuck. A run allowed 10 breaks that finishes on
9 is a 471/480.

This port briefly dropped the 239 entry, on the theory that allowing 10 breaks capped the best
possible result at 470. That was wrong, and it also cost reachability rather than trading it:
because the array counts entries `<= i`, the 9- and 10-entry schedules are identical at every
depth below 239 and the 10-entry one is strictly more permissive above it, so the 10-break search
tree is a *superset*. A leave-one-out sweep had measured the 9-entry variant reaching depth 248
about 70x more rarely (0.2% vs 14.1%).

## Layout

```
SolveBlackwoodKernel.cu              the GPU search kernel
build-kernel.ps1                     nvcc wrapper producing the .ptx
pieces.csv                           the 256-piece set (TheSil numbering)
src/main/resources/
  JBlackwood_Pieces.txt              the same set in Blackwood's own numbering
src/main/java/dk/puzzle/
  blackwood/BlackwoodGpuRunner       main entry point: launch loop, harvesting, saving
  blackwood/BlackwoodSolver          the CPU port; here it builds the candidate tables
  blackwood/BwUtil, BwPiece, ...     Blackwood's tables, break schedule, board encoding
  blackwood/BwGpuTables              flattens the tables into GPU (CSR) form
  gpu/BlackwoodGpuEngine             JCuda host side: upload, launch, read back
  tools/HoleSolver                   exact MRV completion + scoring of partial boards
  io/drive/                          optional Google Drive mirroring
```

## Credit and licence

The search algorithm, candidate tables, break schedule and heuristic gate are
**Joshua Blackwood's** work, from
[jblackwood345/EternityII_Solver](https://github.com/jblackwood345/EternityII_Solver)
(GPL-3.0). This repository is a derivative work: a port of that algorithm to CUDA plus the
tooling around it. See `LICENSE`.
