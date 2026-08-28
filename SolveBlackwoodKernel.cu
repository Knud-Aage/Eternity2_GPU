/**
 * SolveBlackwoodKernel.cu
 *
 * A genuinely faithful GPU-native port of Joshua Blackwood's own algorithm
 * (github.com/jblackwood345/EternityII_Solver) -- NOT a modification of
 * SolveEternityKernel.cu's solvePBP. That kernel stays untouched; this is a
 * new, separate, additive kernel, built so a real three-way comparison is
 * possible: CPU-only (dk.puzzle.blackwood.BlackwoodSolver), the existing
 * generic-index hybrid (solvePBP), and this.
 *
 * Faithfully mirrors dk.puzzle.blackwood.BlackwoodSolver.solvePuzzle() (the
 * already-verified CPU port) -- see that class and BwUtil.java for the
 * reference algorithm being translated here. Candidate tables are flattened
 * host-side by dk.puzzle.blackwood.BwGpuTables from prepare()'s own
 * already-verified table objects (not re-derived a second time) into a CSR
 * (offset+count index, in __constant__ memory) + flat payload (in global
 * device memory) layout.
 *
 * IMPORTANT: this kernel deliberately has NO runtime 4-direction border/
 * mismatch check (unlike solvePBP's matches()/matchKind()) -- Blackwood's
 * real algorithm doesn't have one either. Border-colour correctness falls
 * out of hard rotation filters already baked into table construction (see
 * BlackwoodSolver.prepare(): bottomSides->rotation 0 only, leftSides->
 * rotation 1 only, topSides->rotation 2 only, rightSides*->rotation 3 only,
 * start->rotation 2 only -- confirmed directly against that code before this
 * kernel was written). Porting matches()/matchKind() here would be
 * UNFAITHFUL, not just redundant -- do not "fix" this back in later.
 *
 * All colours are Blackwood's own raw 0-22 numbering throughout, exactly
 * like the CPU port -- never TheSil numbering, no translation needed here.
 *
 * Candidate packing (single int32, mirrors this project's existing
 * getNorth/getEast-style bit packing): byte0 = pieceNumber-1 (0-255),
 * byte1 = topSide (0-22), byte2 = rightSide (0-22), byte3 = rotation
 * (bits0-1) | breakCount (bit2, 0 or 1 -- never 2, addCandidateIfValid
 * already filters those out) | heuristicSideCount (bits3-5, 0-4). Empty-cell
 * sentinel is -1, matching SolveEternityKernel.cu's own convention.
 *
 * 2026-08-04: PERSISTENT PER-THREAD STATE ACROSS LAUNCHES. Originally every
 * launch (~900ms, capped by Windows' WDDM TDR watchdog) started every thread
 * completely fresh at step 0 with a brand-new random corner, and discarded
 * all in-progress search state at the end regardless of how deep it had
 * gotten -- confirmed via nodesTaken == numThreads * stepBudget in the logs,
 * meaning essentially every thread was hitting the per-launch node budget
 * mid-search, never genuine exhaustion (BlackwoodSolver.attemptExhausted()).
 * That capped the deepest possible chronological-backtracking search any
 * single thread could ever accumulate at ~100,000 nodes, no matter how long
 * the process ran in wall-clock time -- and this algorithm's real depth
 * comes from sustained, patient backtracking (confirmed by the CPU port,
 * whose few threads have no such cap and reliably out-depth the GPU's much
 * higher raw node throughput).
 *
 * Now each thread's full search state (board, resume cursors, cumulative
 * break/heuristic bookkeeping, piece-used bits, its own re-randomized
 * bottomSides table, RNG state, current depth, and its personal best-ever
 * board this epoch) is checkpointed to global memory at the end of every
 * launch and reloaded at the start of the next one (see d_needsInit and the
 * d_persist* buffers) -- so a thread's search now genuinely continues across
 * launches instead of restarting, bounded only by how many launches occur
 * within one "epoch" (see BlackwoodGpuRunner.EPOCH_LAUNCHES), not by a single
 * launch's node budget. A thread that reaches genuine exhaustion mid-launch
 * now immediately reseeds a fresh attempt and keeps using its remaining node
 * budget (via seedFreshAttempt()) rather than idling for the rest of the
 * launch -- closing a second gap flagged in the original design as a
 * deliberate de-risking simplification, revisited now that the kernel's
 * correctness is already established.
 *
 * 2026-08-18: SHARED-MEMORY CACHING FOR THE FOUR PER-STEP TABLES. The
 * divergence profile (BlackwoodGpuProfileHarness, see BW_PROFILE_COUNTERS
 * below) measured 100% warp efficiency -- divergence was never the
 * bottleneck. But c_stepToTableId/c_stepBoardIdx/c_breakArray/
 * c_heuristicArray (4KB total) are read from __constant__ memory at a
 * per-lane-different index on every single outer-loop iteration, and this
 * kernel was the only one of the two in this project using 0 bytes of
 * __shared__ memory (confirmed via nvcc -cubin -Xptxas -v) -- solvePBP's own
 * buildSharedIndex() (SolveEternityKernel.cu) already proves this exact
 * pattern pays off in this codebase. The per-thread search body is now
 * factored into runBlackwoodDfsBody() so both entry points below share it
 * verbatim; they differ only in whether the four tables passed in point at
 * __constant__ memory (solveBlackwoodDfs, unchanged behaviour, default) or a
 * one-time-per-block __shared__ copy (solveBlackwoodDfsShared, opt-in via
 * BlackwoodGpuEngine.sharedCacheEnabled). blockDim.x is fixed at 256
 * (BlackwoodGpuEngine's own launch config), matching each table's 256-entry
 * size exactly, so every thread in the block copies exactly one element --
 * no thread-0-only serial copy, no loop needed.
 */

#define NUM_TABLES 10
#define KEY_SPACE  529
#define TABLE_CORNERS         0
#define TABLE_LEFT_SIDES      1
#define TABLE_TOP_SIDES       2
#define TABLE_RIGHT_NOBREAK   3
#define TABLE_RIGHT_BREAK     4
#define TABLE_MIDDLES_NOBREAK 5
#define TABLE_MIDDLES_BREAK   6
#define TABLE_SOUTH_START     7
#define TABLE_WEST_START      8
#define TABLE_START           9

#define FIRST_BREAK_INDEX   201
#define HEURISTIC_MAX_INDEX 160
#define MAX_BOTTOM_PAYLOAD  96

// Warp-divergence instrumentation, compiled in ONLY with -DBW_PROFILE_COUNTERS
// (see build-blackwood-profile-ptx.ps1). Without the flag this file produces
// exactly the PTX it always has -- same signature, same body, zero cost -- so
// the production kernel is unaffected by anything below.
//
// Why this exists: the GPU port has always trailed the CPU port of the same
// algorithm (12 depth records logged vs 1785/1066 for the two CPU variants),
// and the usual explanation offered for GPU backtracking search is warp
// divergence -- but nobody had ever measured it here. These counters answer
// three specific competing hypotheses with numbers instead of plausibility:
// (1) BW_PC_ACTIVE_LANE_SUM/WARP_ITERATIONS = mean active lanes per warp per
//     search step. 32.0 means no divergence at all; ~3 would be the "GPU
//     running at 10% efficiency" claim. This is the headline number.
// (2) The RESEED_* counters test the narrower theory that one lane hitting
//     genuine exhaustion stalls 31 masked-off warp-mates through
//     seedFreshAttempt()'s insertion sort.
// (3) The CNT_* counters record the candidate-list length distribution --
//     a precondition for whether uniform-trip-count restructuring could ever
//     pay off (it only helps if lengths cluster tightly).
#define BW_PC_WARP_ITERATIONS       0
#define BW_PC_ACTIVE_LANE_SUM       1
#define BW_PC_GENERAL_WARP_SAMPLES  2
#define BW_PC_GENERAL_MIXED         3
#define BW_PC_RESEED_WARP_SAMPLES   4
#define BW_PC_RESEED_MIXED          5
#define BW_PC_RESEED_EVENTS         6
#define BW_PC_CNT_SUM               7
#define BW_PC_CNT_SAMPLES           8
#define BW_PC_CNT_MAX               9
#define BW_PC_SLOTS                16

__constant__ int c_csrOffset[NUM_TABLES * KEY_SPACE];        // 21,160 B
__constant__ int c_csrCount [NUM_TABLES * KEY_SPACE];        // 21,160 B
__constant__ int c_bottomRawOffset[23];                      // 92 B
__constant__ int c_bottomRawCount [23];                      // 92 B
__constant__ int c_bottomRawPayload[MAX_BOTTOM_PAYLOAD];     // 384 B, real count ~56
__constant__ int c_stepToTableId[256];                       // 1,024 B; row-0 steps hold an unused sentinel
__constant__ int c_stepBoardIdx[256];                        // 1,024 B; row*16+col per step
__constant__ int c_breakArray[256];                          // 1,024 B
__constant__ int c_heuristicArray[256];                      // 1,024 B
// Total ~47 KB of this module's own fresh 64 KB __constant__ budget.

__device__ inline int bwPieceNum(int r)       { return (r & 0xFF) + 1; }
__device__ inline int bwTopSide(int r)        { return (r >> 8)  & 0xFF; }
__device__ inline int bwRightSide(int r)      { return (r >> 16) & 0xFF; }
__device__ inline int bwRotation(int r)       { return (r >> 24) & 0x3; }
__device__ inline int bwBreakCount(int r)     { return (r >> 26) & 0x1; }
__device__ inline int bwHeuristicCount(int r) { return (r >> 27) & 0x7; }

// Packed-bit helpers for the 256-entry pieceUsed flag -- copied verbatim from
// SolveEternityKernel.cu (__device__ functions aren't shared across
// separately-compiled modules).
__device__ inline bool bitGet(const unsigned int* bits, int idx) {
    return (bits[idx >> 5] >> (idx & 31)) & 1u;
}
__device__ inline void bitSet(unsigned int* bits, int idx) {
    bits[idx >> 5] |= (1u << (idx & 31));
}
__device__ inline void bitClear(unsigned int* bits, int idx) {
    bits[idx >> 5] &= ~(1u << (idx & 31));
}

// xorshift64* -- lightweight in-kernel PRNG. No curand infrastructure exists
// anywhere in this codebase (dependency present in pom.xml/cp.txt, zero
// actual device-side usage); adding it for one kernel's candidate-order
// jitter isn't worth the new dependency surface. This mirrors the
// DISTRIBUTION SHAPE of BwUtil's java.util.Random-based jitter (uniform
// picks, same score formulas, same re-randomization cadence) -- NOT
// java.util.Random's exact algorithm or draw order. A GPU thread and a CPU
// attempt given "the same seed" will legitimately produce different boards;
// that's expected, not a bug.
__device__ inline unsigned long long xorshift64star(unsigned long long *state) {
    unsigned long long x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

// Uniform int in [0, bound) -- mirrors java.util.Random.nextInt(bound)'s
// CONTRACT (uniform over the range), not its rejection-sampling algorithm.
__device__ inline int randInt(unsigned long long *state, unsigned int bound) {
    if (bound == 0) return 0;
    return (int)(xorshift64star(state) % (unsigned long long)bound);
}

// Seeds a brand-new attempt in place: rebuilds this thread's own bottomSides
// (per-attempt re-randomization, matching Blackwood's own cadence for that
// one table), then picks a uniform-random corner for step 0. Used both for a
// thread's very first attempt (d_needsInit) and for re-seeding immediately
// after genuine exhaustion mid-launch (solveIndex < 1) -- factored out so
// both call sites can never drift apart.
// Clears the board and rebuilds this thread's own randomized bottomSides table -- the part
// every new attempt needs regardless of whether step 0 onward then comes from a random corner
// (seedFreshAttempt) or from a saved board (seedFromSavedBoard). Split out so those two entry
// points can never drift apart on the setup they share.
__device__ inline void resetAndBuildBottomSides(
    int* board, int* pieceIndexToTryNext,
    unsigned int* pieceUsedBits, int* bsOffset, int* bsCount, int* bsPayload,
    unsigned long long* rngState)
{
    for (int i = 0; i < 256; i++) { board[i] = -1; pieceIndexToTryNext[i] = 0; }
    for (int i = 0; i < 8; i++) pieceUsedBits[i] = 0;

    // Mirrors BwUtil.sortAndFreezeBottomSides's formula exactly, applied to
    // the same raw pool BwGpuTables.build() already extracted --
    // (heuristicCount>0 ? 100 : 0) + jitter, descending, insertion sort.
    {
        int keys[MAX_BOTTOM_PAYLOAD]; // scratch, reused per bucket
        int fillPos = 0;
        for (int left = 0; left < 23; left++) {
            int rawOff = c_bottomRawOffset[left];
            int rawCnt = c_bottomRawCount[left];
            bsOffset[left] = fillPos;
            bsCount[left] = rawCnt;
            for (int i = 0; i < rawCnt; i++) {
                int rec = c_bottomRawPayload[rawOff + i];
                int hc = bwHeuristicCount(rec);
                keys[i] = (hc > 0 ? 100 : 0) + randInt(rngState, 99);
                bsPayload[fillPos + i] = rec;
            }
            for (int i = 1; i < rawCnt; i++) {
                int keyVal = keys[i];
                int recVal = bsPayload[fillPos + i];
                int j = i - 1;
                while (j >= 0 && keys[j] < keyVal) {
                    keys[j + 1] = keys[j];
                    bsPayload[fillPos + j + 1] = bsPayload[fillPos + j];
                    j--;
                }
                keys[j + 1] = keyVal;
                bsPayload[fillPos + j + 1] = recVal;
            }
            fillPos += rawCnt;
        }
    }
}

__device__ inline void seedFreshAttempt(
    const int* d_payload,
    int* board, int* pieceIndexToTryNext, int* cumulativeBreaks, int* cumulativeHeuristicSideCount,
    unsigned int* pieceUsedBits, int* bsOffset, int* bsCount, int* bsPayload,
    unsigned long long* rngState)
{
    resetAndBuildBottomSides(board, pieceIndexToTryNext, pieceUsedBits,
                             bsOffset, bsCount, bsPayload, rngState);

    // Step 0: uniform-random pick from corners at key 0 (left=0,bottom=0) --
    // mirrors BlackwoodSolver.solvePuzzle()'s uniform pick from corners[0]
    // exactly. corners[0] non-empty is a verified invariant (BlackwoodSolver.
    // prepare() throws if not; BwGpuTablesTest asserts the same survives CSR
    // flattening), so cnt>0 here is not runtime-checked.
    int off = c_csrOffset[TABLE_CORNERS * KEY_SPACE + 0];
    int cnt = c_csrCount [TABLE_CORNERS * KEY_SPACE + 0];
    int pick = off + randInt(rngState, (unsigned int)cnt);
    board[0] = d_payload[pick];
    bitSet(pieceUsedBits, bwPieceNum(board[0]) - 1);
    cumulativeBreaks[0] = 0;
    cumulativeHeuristicSideCount[0] = bwHeuristicCount(board[0]);
}

/**
 * Resolves the candidate list for one step exactly the way the main search loop does, so a
 * replayed board is indexed against the identical tables the search will use when it later
 * backtracks through those same steps. Any divergence here would corrupt the resume cursors.
 *
 * @param stepToTableId either c_stepToTableId directly (__constant__) or a block's own
 *                       __shared__ copy of it -- the caller decides which; this function is
 *                       agnostic to where the 256 ints actually live.
 */
__device__ inline void candidateListForStep(
    int solveIndex, int boardIdx, const int* board,
    const int* bsOffset, const int* bsCount, const int* stepToTableId,
    int* outOff, int* outCnt, bool* outUseBottom)
{
    int row = boardIdx >> 4;
    int col = boardIdx & 15;
    *outUseBottom = false;
    if (row == 0) {
        int westRight = bwRightSide(board[boardIdx - 1]);
        if (col < 15) {
            *outUseBottom = true;
            *outOff = bsOffset[westRight];
            *outCnt = bsCount[westRight];
        } else {
            int key = westRight * 23;
            *outOff = c_csrOffset[TABLE_CORNERS * KEY_SPACE + key];
            *outCnt = c_csrCount [TABLE_CORNERS * KEY_SPACE + key];
        }
    } else {
        int leftSide = (col == 0) ? 0 : bwRightSide(board[boardIdx - 1]);
        int southTop = bwTopSide(board[boardIdx - 16]);
        int key = leftSide * 23 + southTop;
        int tableId = stepToTableId[solveIndex];
        *outOff = c_csrOffset[tableId * KEY_SPACE + key];
        *outCnt = c_csrCount [tableId * KEY_SPACE + key];
    }
}

/**
 * Rebuilds a full in-progress search state from a previously saved board, so the GPU can start
 * at the frontier instead of re-deriving 250 pieces of already-known progress from scratch.
 *
 * <p>Crucially this does NOT just copy pieces onto the board -- a copied board would have no
 * resume cursors, so the first backtrack would re-try pieces the original search already
 * rejected. Instead each seeded piece is LOCATED in its step's candidate list, and
 * pieceIndexToTryNext[step] is set past it. Backtracking into seeded territory then correctly
 * continues with the alternatives that were never tried, which is what makes the seeded state a
 * genuine resumption rather than a snapshot.</p>
 *
 * <p>Candidate ORDER is re-randomized per epoch while candidate MEMBERSHIP is not, so searching
 * for the piece (rather than trusting a stored index) is what makes a seed survive table
 * rebuilds.</p>
 *
 * @param seed  per-step (pieceNumber << 2) | rotation, or negative where the seed ends
 * @param stepBoardIdx/stepToTableId see candidateListForStep's own note -- __constant__ or
 *                                   __shared__, this function doesn't care which
 * @return number of pieces successfully placed. A short return is safe, not an error: the thread
 *         simply continues from the shallower point it did manage to reach.
 */
__device__ inline int seedFromSavedBoard(
    const int* d_payload, const int* seed, int targetDepth,
    int* board, int* pieceIndexToTryNext, int* cumulativeBreaks, int* cumulativeHeuristicSideCount,
    unsigned int* pieceUsedBits, int* bsOffset, int* bsCount, int* bsPayload,
    const int* stepBoardIdx, const int* stepToTableId,
    unsigned long long* rngState)
{
    resetAndBuildBottomSides(board, pieceIndexToTryNext, pieceUsedBits,
                             bsOffset, bsCount, bsPayload, rngState);

    for (int step = 0; step <= targetDepth; step++) {
        int want = seed[step];
        if (want < 0) return step;                 // seed ran out -- continue from here
        int wantPiece = want >> 2;
        int wantRot = want & 3;
        if (wantPiece < 1 || wantPiece > 256) return step;
        if (bitGet(pieceUsedBits, wantPiece - 1)) return step;  // duplicate piece: refuse to corrupt state

        int boardIdx = stepBoardIdx[step];
        int off, cnt;
        bool useBottom = false;
        if (step == 0) {
            off = c_csrOffset[TABLE_CORNERS * KEY_SPACE + 0];
            cnt = c_csrCount [TABLE_CORNERS * KEY_SPACE + 0];
        } else {
            candidateListForStep(step, boardIdx, board, bsOffset, bsCount, stepToTableId, &off, &cnt, &useBottom);
        }

        int foundAt = -1;
        int foundRec = 0;
        for (int i = 0; i < cnt; i++) {
            int rec = useBottom ? bsPayload[off + i] : d_payload[off + i];
            if (bwPieceNum(rec) == wantPiece && bwRotation(rec) == wantRot) { foundAt = i; foundRec = rec; break; }
        }
        if (foundAt < 0) return step;              // not reachable through these tables -- stop cleanly

        board[boardIdx] = foundRec;
        bitSet(pieceUsedBits, wantPiece - 1);
        int prevBreaks = (step == 0) ? 0 : cumulativeBreaks[step - 1];
        int prevHeur   = (step == 0) ? 0 : cumulativeHeuristicSideCount[step - 1];
        cumulativeBreaks[step] = prevBreaks + ((step == 0) ? 0 : bwBreakCount(foundRec));
        cumulativeHeuristicSideCount[step] = prevHeur + bwHeuristicCount(foundRec);
        pieceIndexToTryNext[step] = foundAt + 1;   // resume AFTER the seeded piece
    }
    return targetDepth + 1;
}

/**
 * Starts a new attempt, from a saved board when seeding is on and from a random corner otherwise.
 * Both the very first attempt and every mid-launch restart after genuine exhaustion go through
 * here, so a seeded run never silently degrades into unseeded work once threads exhaust.
 *
 * <p>freshFractionPercent reserves that percentage of attempts for genuine from-scratch search
 * even while seeds are loaded. Without it, EVERY attempt resumes one of a small set of already
 * heavily-searched archive boards and only re-explores its last maxRetreat steps -- so the whole
 * population stays confined to variations of boards that are plausibly already exhausted, with no
 * mechanism to look anywhere else. The fresh fraction is what feeds genuinely new boards back into
 * the seed pool at the next epoch, making the run a real explore/exploit loop rather than a closed
 * one. Affordable because unseeded search is not weak in absolute terms: after the epoch-reset fix
 * it reaches ~249 pieces in 90 seconds on its own.</p>
 *
 * @return the new solveIndex (== number of pieces now placed)
 */
__device__ inline int startNewAttempt(
    const int* d_payload, const int* d_seedBoards, const int* d_seedDepths,
    int numSeeds, int maxRetreat, int freshFractionPercent, unsigned int* d_seedShortfalls,
    int* board, int* pieceIndexToTryNext, int* cumulativeBreaks, int* cumulativeHeuristicSideCount,
    unsigned int* pieceUsedBits, int* bsOffset, int* bsCount, int* bsPayload,
    const int* stepBoardIdx, const int* stepToTableId,
    unsigned long long* rngState)
{
    bool goFresh = (freshFractionPercent > 0) && (randInt(rngState, 100) < (unsigned int)freshFractionPercent);
    if (numSeeds > 0 && !goFresh) {
        // Spread threads over both WHICH saved board they resume and HOW FAR BACK they pull from
        // its tip. Without that spread, threads sharing a seed would walk identical candidate
        // orders and duplicate each other's work -- candidate order is global, only bottomSides
        // is per-thread.
        int seedIdx = randInt(rngState, (unsigned int)numSeeds);
        int fullDepth = d_seedDepths[seedIdx];
        int target = fullDepth - 1 - ((maxRetreat > 0) ? randInt(rngState, (unsigned int)(maxRetreat + 1)) : 0);
        if (target < 0) target = 0;

        int placed = seedFromSavedBoard(d_payload, d_seedBoards + (size_t)seedIdx * 256, target,
                                        board, pieceIndexToTryNext, cumulativeBreaks,
                                        cumulativeHeuristicSideCount, pieceUsedBits,
                                        bsOffset, bsCount, bsPayload, stepBoardIdx, stepToTableId, rngState);
        if (placed < target + 1) atomicAdd(d_seedShortfalls, 1u);
        if (placed >= 1) return placed;
        // Could not place even step 0 -- fall through rather than run on a broken state.
    }
    seedFreshAttempt(d_payload, board, pieceIndexToTryNext, cumulativeBreaks,
                      cumulativeHeuristicSideCount, pieceUsedBits, bsOffset, bsCount, bsPayload, rngState);
    return 1;
}

/**
 * The full per-thread search body, shared verbatim by both __global__ entry points below. Takes
 * the four hot per-step tables (stepToTableId/stepBoardIdx/breakArray/heuristicArray) as plain
 * pointers rather than reaching for the c_* __constant__ globals directly, so the exact same
 * logic runs unchanged whether the caller passes __constant__ memory (solveBlackwoodDfs) or a
 * block-local __shared__ copy of it (solveBlackwoodDfsShared) -- see the 2026-08-18 header note.
 */
__device__ inline void runBlackwoodDfsBody(
    int tid,
    const int* d_payload,             // global memory: flat candidate payload, all 10 tables concatenated
    unsigned long long seedBase,      // host-varied every launch (nanoTime ^ launchCounter)
    unsigned long long stepBudget,    // per-thread node cap THIS launch -- a TDR safety valve, not a search-depth cap
    int* d_gpuHighScore,              // atomic high-water maxSolveIndex across all threads, all launches this run
    int* d_bestBoardOut,              // [256] packed records of the current best board
    int* d_solution,                  // [256] set once if any thread reaches step 256 (a genuine full solve)
    int* d_solvedFlag,
    unsigned long long* d_totalNodes, // atomicAdd, for throughput reporting
    int* d_threadDepths,              // [numThreads] this thread's maxSolveIndex this attempt
    // Persistent per-thread search state (global memory, survives across launches within one
    // epoch -- see BlackwoodGpuRunner.EPOCH_LAUNCHES). Each buffer is indexed [tid * perThreadSize + i].
    int* d_persistBoard,
    int* d_persistPieceIndexToTryNext,
    int* d_persistCumulativeBreaks,
    int* d_persistCumulativeHeuristicSideCount,
    unsigned int* d_persistPieceUsedBits,
    int* d_persistBsOffset,
    int* d_persistBsCount,
    int* d_persistBsPayload,
    unsigned long long* d_persistRngState,
    int* d_persistSolveIndex,
    int* d_persistBestBoard,
    int* d_persistBestPiecesPlaced,
    int* d_needsInit,                 // [numThreads] 1 = start a fresh attempt this launch, 0 = resume persisted state
    // Seeding from previously saved deep boards. numSeeds == 0 reproduces the original
    // always-start-from-a-random-corner behaviour exactly, so this is opt-in.
    const int* d_seedBoards,          // [numSeeds * 256], per step: (pieceNumber << 2) | rotation, negative = end
    const int* d_seedDepths,          // [numSeeds] how many steps each seed actually covers
    int numSeeds,
    int maxRetreat,                   // per-thread random pull-back from the seed's full depth (diversity)
    int freshFractionPercent,         // % of attempts that ignore seeds entirely and start from a random corner
    unsigned int* d_seedShortfalls,   // counts threads whose replay stopped short of its target
    const int* stepToTableId, const int* stepBoardIdx, const int* breakArray, const int* heuristicArray
#ifdef BW_PROFILE_COUNTERS
    , unsigned long long* d_profileCounters   // [BW_PC_SLOTS]
#endif
)
{
#ifdef BW_PROFILE_COUNTERS
    const int bwLane = threadIdx.x & 31;
#endif

    int board[256];
    int pieceIndexToTryNext[256];
    int cumulativeBreaks[256];
    int cumulativeHeuristicSideCount[256];
    unsigned int pieceUsedBits[8];
    int bsOffset[23];
    int bsCount[23];
    int bsPayload[MAX_BOTTOM_PAYLOAD];
    int bestLocalBoard[256];

    unsigned long long rngState;
    int solveIndex;
    int maxSolveIndex;
    int bestPiecesPlaced;

    const int P256 = tid * 256;
    const int P23  = tid * 23;
    const int P96  = tid * MAX_BOTTOM_PAYLOAD;
    const int P8   = tid * 8;

    if (d_needsInit[tid]) {
        rngState = seedBase ^ ((unsigned long long)tid * 0x9E3779B97F4A7C15ULL);
        if (rngState == 0) rngState = 0x9E3779B97F4A7C15ULL; // xorshift64* requires a non-zero state

        solveIndex = startNewAttempt(d_payload, d_seedBoards, d_seedDepths, numSeeds, maxRetreat, freshFractionPercent,
                                     d_seedShortfalls, board, pieceIndexToTryNext, cumulativeBreaks,
                                     cumulativeHeuristicSideCount, pieceUsedBits,
                                     bsOffset, bsCount, bsPayload, stepBoardIdx, stepToTableId, &rngState);
        maxSolveIndex = solveIndex;
        bestPiecesPlaced = solveIndex;
        for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
        d_needsInit[tid] = 0;
    } else {
        for (int i = 0; i < 256; i++) {
            board[i] = d_persistBoard[P256 + i];
            pieceIndexToTryNext[i] = d_persistPieceIndexToTryNext[P256 + i];
            cumulativeBreaks[i] = d_persistCumulativeBreaks[P256 + i];
            cumulativeHeuristicSideCount[i] = d_persistCumulativeHeuristicSideCount[P256 + i];
            bestLocalBoard[i] = d_persistBestBoard[P256 + i];
        }
        for (int i = 0; i < 8; i++) pieceUsedBits[i] = d_persistPieceUsedBits[P8 + i];
        for (int i = 0; i < 23; i++) {
            bsOffset[i] = d_persistBsOffset[P23 + i];
            bsCount[i] = d_persistBsCount[P23 + i];
        }
        for (int i = 0; i < MAX_BOTTOM_PAYLOAD; i++) bsPayload[i] = d_persistBsPayload[P96 + i];
        rngState = d_persistRngState[tid];
        solveIndex = d_persistSolveIndex[tid];
        bestPiecesPlaced = d_persistBestPiecesPlaced[tid];
        maxSolveIndex = bestPiecesPlaced; // these two are always kept equal -- see the update block below
    }

    unsigned long long nodeCount = 0;
    bool completed = false;

    while (true) {
        nodeCount++;

#ifdef BW_PROFILE_COUNTERS
        // THE headline measurement: how many lanes of this warp are still
        // executing this search step at all. Threads leave this loop at
        // wildly different times (budget exhaustion, a peer solving, genuine
        // completion), so __activemask() -- NOT a hardcoded 0xFFFFFFFF mask --
        // is the only safe way to ballot here; naming already-exited threads
        // in a sync mask is undefined behaviour and can hang the kernel.
        // Only the lowest active lane does the atomicAdd, so this costs 1/32
        // of the atomic traffic a naive per-lane version would.
        {
            unsigned act = __activemask();
            if (bwLane == __ffs(act) - 1) {
                atomicAdd(&d_profileCounters[BW_PC_WARP_ITERATIONS], 1ULL);
                atomicAdd(&d_profileCounters[BW_PC_ACTIVE_LANE_SUM], (unsigned long long)__popc(act));
            }
        }
#endif

        if (solveIndex > maxSolveIndex) {
            maxSolveIndex = solveIndex;
            if (maxSolveIndex > bestPiecesPlaced) {
                bestPiecesPlaced = maxSolveIndex;
                for (int i = 0; i < 256; i++) bestLocalBoard[i] = board[i];
            }
            if (maxSolveIndex >= 256) { completed = true; break; }
        }

        if (nodeCount > stepBudget) break;           // this launch's budget hit -- checkpoint and resume next launch
        if (*d_solvedFlag == 1) break;                // another thread already found a full solution this launch

#ifdef BW_PROFILE_COUNTERS
        // Tests the narrow hypothesis: when only SOME lanes of a warp need a
        // reseed, the others sit masked off while seedFreshAttempt()'s nested
        // loop + 23-bucket insertion sort runs serially. A high MIXED/SAMPLES
        // ratio here is what would justify building a warp-cooperative reseed.
        {
            unsigned act = __activemask();
            unsigned needs = __ballot_sync(act, solveIndex < 1);
            if (bwLane == __ffs(act) - 1) {
                int yes = __popc(needs), tot = __popc(act);
                atomicAdd(&d_profileCounters[BW_PC_RESEED_WARP_SAMPLES], 1ULL);
                atomicAdd(&d_profileCounters[BW_PC_RESEED_EVENTS], (unsigned long long)yes);
                if (yes != 0 && yes != tot) atomicAdd(&d_profileCounters[BW_PC_RESEED_MIXED], 1ULL);
            }
        }
#endif

        if (solveIndex < 1) {
            // Genuine exhaustion of this attempt (BlackwoodSolver.attemptExhausted()'s guard).
            // Rather than idling for the rest of this launch's node budget, start another attempt
            // immediately and keep going -- from a saved board if seeding is on (a thread that
            // exhausts a deep subtree should return to the frontier, not to a random corner).
            solveIndex = startNewAttempt(d_payload, d_seedBoards, d_seedDepths, numSeeds, maxRetreat, freshFractionPercent,
                                         d_seedShortfalls, board, pieceIndexToTryNext, cumulativeBreaks,
                                         cumulativeHeuristicSideCount, pieceUsedBits,
                                         bsOffset, bsCount, bsPayload, stepBoardIdx, stepToTableId, &rngState);
            continue;
        }

        int boardIdx = stepBoardIdx[solveIndex];

        if (board[boardIdx] != -1) {
            bitClear(pieceUsedBits, bwPieceNum(board[boardIdx]) - 1);
            board[boardIdx] = -1;
        }

        // Shared with seedFromSavedBoard() so a replayed board's resume cursors are guaranteed to
        // index the same lists this loop will scan. (row=0,col=0) is exclusively step 0, which the
        // solveIndex<1 guard above keeps this loop from ever revisiting, so the helper's row-0
        // branch always has a valid already-placed west neighbour.
        int off, cnt;
        bool useBottom = false;
        candidateListForStep(solveIndex, boardIdx, board, bsOffset, bsCount, stepToTableId, &off, &cnt, &useBottom);

#ifdef BW_PROFILE_COUNTERS
        // General mixed-warp fraction (do the lanes of this warp even agree on
        // whether there's a candidate list to scan?), plus the candidate-list
        // length distribution. __reduce_add_sync/__reduce_max_sync are single
        // instructions on sm_80+ (we target sm_120), so the true per-lane sum
        // and max cost about the same as sampling one lane would.
        {
            unsigned act = __activemask();
            unsigned hasCandidates = __ballot_sync(act, cnt > 0);
            unsigned sumCnt = __reduce_add_sync(act, (unsigned)cnt);
            unsigned maxCnt = __reduce_max_sync(act, (unsigned)cnt);
            if (bwLane == __ffs(act) - 1) {
                int yes = __popc(hasCandidates), tot = __popc(act);
                atomicAdd(&d_profileCounters[BW_PC_GENERAL_WARP_SAMPLES], 1ULL);
                if (yes != 0 && yes != tot) atomicAdd(&d_profileCounters[BW_PC_GENERAL_MIXED], 1ULL);
                atomicAdd(&d_profileCounters[BW_PC_CNT_SUM], (unsigned long long)sumCnt);
                atomicAdd(&d_profileCounters[BW_PC_CNT_SAMPLES], (unsigned long long)tot);
                atomicMax(&d_profileCounters[BW_PC_CNT_MAX], (unsigned long long)maxCnt);
            }
        }
#endif

        bool foundPiece = false;
        if (cnt > 0) {
            int breaksThisTurn = breakArray[solveIndex] - cumulativeBreaks[solveIndex - 1];
            bool heuristicGateActive = (solveIndex <= HEURISTIC_MAX_INDEX);
            int heuristicFloor = heuristicGateActive ? heuristicArray[solveIndex] : 0;

            for (int i = pieceIndexToTryNext[solveIndex]; i < cnt; i++) {
                int rec = useBottom ? bsPayload[off + i] : d_payload[off + i];
                if (bwBreakCount(rec) > breaksThisTurn) break; // sort-order invariant: table is break-count-monotonic

                int pieceNum = bwPieceNum(rec);
                if (bitGet(pieceUsedBits, pieceNum - 1)) continue;

                int hc = bwHeuristicCount(rec);
                if (heuristicGateActive && (cumulativeHeuristicSideCount[solveIndex - 1] + hc) < heuristicFloor) {
                    break; // abandons the WHOLE scan for this step, not just this candidate -- matches the CPU port exactly
                }

                board[boardIdx] = rec;
                bitSet(pieceUsedBits, pieceNum - 1);
                cumulativeBreaks[solveIndex] = cumulativeBreaks[solveIndex - 1] + bwBreakCount(rec);
                cumulativeHeuristicSideCount[solveIndex] = cumulativeHeuristicSideCount[solveIndex - 1] + hc;
                pieceIndexToTryNext[solveIndex] = i + 1;
                foundPiece = true;
                solveIndex++;
                break;
            }
        }

        if (!foundPiece) {
            pieceIndexToTryNext[solveIndex] = 0;
            solveIndex--;
        }
    }

    if (completed) {
        if (atomicExch(d_solvedFlag, 1) == 0) {
            for (int i = 0; i < 256; i++) d_solution[i] = board[i];
        }
        // A solved board has nowhere further to search from -- start fresh next launch.
        d_needsInit[tid] = 1;
    } else {
        // Checkpoint this thread's in-progress state so the next launch resumes it instead of
        // discarding it -- the core fix this file exists for (see the 2026-08-04 header note).
        for (int i = 0; i < 256; i++) {
            d_persistBoard[P256 + i] = board[i];
            d_persistPieceIndexToTryNext[P256 + i] = pieceIndexToTryNext[i];
            d_persistCumulativeBreaks[P256 + i] = cumulativeBreaks[i];
            d_persistCumulativeHeuristicSideCount[P256 + i] = cumulativeHeuristicSideCount[i];
            d_persistBestBoard[P256 + i] = bestLocalBoard[i];
        }
        for (int i = 0; i < 8; i++) d_persistPieceUsedBits[P8 + i] = pieceUsedBits[i];
        for (int i = 0; i < 23; i++) {
            d_persistBsOffset[P23 + i] = bsOffset[i];
            d_persistBsCount[P23 + i] = bsCount[i];
        }
        for (int i = 0; i < MAX_BOTTOM_PAYLOAD; i++) d_persistBsPayload[P96 + i] = bsPayload[i];
        d_persistRngState[tid] = rngState;
        d_persistSolveIndex[tid] = solveIndex;
        d_persistBestPiecesPlaced[tid] = bestPiecesPlaced;
        // d_needsInit[tid] is already 0 (cleared above on init, or was already 0 on a prior resume).
    }

    // Same lock-bit atomic best-board update pattern as solvePBP (SolveEternityKernel.cu).
    int globalMaxRaw = *d_gpuHighScore;
    int globalMax    = globalMaxRaw & 0x0FFFFFFF;
    while (bestPiecesPlaced > globalMax) {
        int expected  = globalMax;
        int lockedVal = bestPiecesPlaced | 0x40000000;
        int oldVal    = atomicCAS(d_gpuHighScore, expected, lockedVal);
        if (oldVal == expected) {
            for (int i = 0; i < 256; i++) d_bestBoardOut[i] = bestLocalBoard[i];
            __threadfence();
            atomicExch(d_gpuHighScore, bestPiecesPlaced);
            break;
        }
        globalMaxRaw = oldVal;
        globalMax    = globalMaxRaw & 0x0FFFFFFF;
    }
    atomicAdd(d_totalNodes, nodeCount);
    d_threadDepths[tid] = bestPiecesPlaced;
}

// Unchanged behaviour: the four hot tables are read straight from __constant__ memory, exactly as
// before this file's 2026-08-18 note. Default entry point (BlackwoodGpuEngine.sharedCacheEnabled
// starts false), and always available as a fallback regardless of that flag.
extern "C" __global__ void solveBlackwoodDfs(
    const int* d_payload,
    unsigned long long seedBase,
    unsigned long long stepBudget,
    int  numThreads,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    int* d_solution,
    int* d_solvedFlag,
    unsigned long long* d_totalNodes,
    int* d_threadDepths,
    int* d_persistBoard,
    int* d_persistPieceIndexToTryNext,
    int* d_persistCumulativeBreaks,
    int* d_persistCumulativeHeuristicSideCount,
    unsigned int* d_persistPieceUsedBits,
    int* d_persistBsOffset,
    int* d_persistBsCount,
    int* d_persistBsPayload,
    unsigned long long* d_persistRngState,
    int* d_persistSolveIndex,
    int* d_persistBestBoard,
    int* d_persistBestPiecesPlaced,
    int* d_needsInit,
    const int* d_seedBoards,
    const int* d_seedDepths,
    int numSeeds,
    int maxRetreat,
    int freshFractionPercent,
    unsigned int* d_seedShortfalls
#ifdef BW_PROFILE_COUNTERS
    , unsigned long long* d_profileCounters
#endif
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numThreads) return;

    runBlackwoodDfsBody(tid, d_payload, seedBase, stepBudget, d_gpuHighScore, d_bestBoardOut, d_solution,
        d_solvedFlag, d_totalNodes, d_threadDepths, d_persistBoard, d_persistPieceIndexToTryNext,
        d_persistCumulativeBreaks, d_persistCumulativeHeuristicSideCount, d_persistPieceUsedBits,
        d_persistBsOffset, d_persistBsCount, d_persistBsPayload, d_persistRngState, d_persistSolveIndex,
        d_persistBestBoard, d_persistBestPiecesPlaced, d_needsInit, d_seedBoards, d_seedDepths,
        numSeeds, maxRetreat, freshFractionPercent, d_seedShortfalls,
        c_stepToTableId, c_stepBoardIdx, c_breakArray, c_heuristicArray
#ifdef BW_PROFILE_COUNTERS
        , d_profileCounters
#endif
    );
}

// 2026-08-18: shared-memory-cached twin of solveBlackwoodDfs -- identical signature and identical
// search logic (runBlackwoodDfsBody), differing only in where the four hot per-step tables live.
// Opt-in via BlackwoodGpuEngine.sharedCacheEnabled; see the header note for why.
extern "C" __global__ void solveBlackwoodDfsShared(
    const int* d_payload,
    unsigned long long seedBase,
    unsigned long long stepBudget,
    int  numThreads,
    int* d_gpuHighScore,
    int* d_bestBoardOut,
    int* d_solution,
    int* d_solvedFlag,
    unsigned long long* d_totalNodes,
    int* d_threadDepths,
    int* d_persistBoard,
    int* d_persistPieceIndexToTryNext,
    int* d_persistCumulativeBreaks,
    int* d_persistCumulativeHeuristicSideCount,
    unsigned int* d_persistPieceUsedBits,
    int* d_persistBsOffset,
    int* d_persistBsCount,
    int* d_persistBsPayload,
    unsigned long long* d_persistRngState,
    int* d_persistSolveIndex,
    int* d_persistBestBoard,
    int* d_persistBestPiecesPlaced,
    int* d_needsInit,
    const int* d_seedBoards,
    const int* d_seedDepths,
    int numSeeds,
    int maxRetreat,
    int freshFractionPercent,
    unsigned int* d_seedShortfalls
#ifdef BW_PROFILE_COUNTERS
    , unsigned long long* d_profileCounters
#endif
)
{
    // 4KB total (4 x 256 x 4 bytes), populated once per block. blockDim.x is fixed at 256
    // (BlackwoodGpuEngine's launch config) -- exactly one element per thread, no loop, no
    // thread-0-only serial copy.
    __shared__ int sm_stepToTableId[256];
    __shared__ int sm_stepBoardIdx[256];
    __shared__ int sm_breakArray[256];
    __shared__ int sm_heuristicArray[256];

    sm_stepToTableId[threadIdx.x]  = c_stepToTableId[threadIdx.x];
    sm_stepBoardIdx[threadIdx.x]   = c_stepBoardIdx[threadIdx.x];
    sm_breakArray[threadIdx.x]     = c_breakArray[threadIdx.x];
    sm_heuristicArray[threadIdx.x] = c_heuristicArray[threadIdx.x];
    __syncthreads();

    // Bounds check AFTER the sync, not before -- every thread in the block must reach
    // __syncthreads() regardless of numThreads, or the block hangs (mirrors
    // SolveEternityKernel.cu's own buildSharedIndex()/__syncthreads() ordering).
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numThreads) return;

    runBlackwoodDfsBody(tid, d_payload, seedBase, stepBudget, d_gpuHighScore, d_bestBoardOut, d_solution,
        d_solvedFlag, d_totalNodes, d_threadDepths, d_persistBoard, d_persistPieceIndexToTryNext,
        d_persistCumulativeBreaks, d_persistCumulativeHeuristicSideCount, d_persistPieceUsedBits,
        d_persistBsOffset, d_persistBsCount, d_persistBsPayload, d_persistRngState, d_persistSolveIndex,
        d_persistBestBoard, d_persistBestPiecesPlaced, d_needsInit, d_seedBoards, d_seedDepths,
        numSeeds, maxRetreat, freshFractionPercent, d_seedShortfalls,
        sm_stepToTableId, sm_stepBoardIdx, sm_breakArray, sm_heuristicArray
#ifdef BW_PROFILE_COUNTERS
        , d_profileCounters
#endif
    );
}
