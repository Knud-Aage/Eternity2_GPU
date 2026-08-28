package dk.puzzle.gpu;

import dk.puzzle.blackwood.BwGpuTables;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import java.util.Arrays;
import java.util.List;

import static jcuda.driver.JCudaDriver.*;

/**
 * GPU engine for the genuinely faithful Blackwood-native kernel ({@code SolveBlackwoodKernel.cu}
 * / {@code solveBlackwoodDfs}) -- deliberately separate from {@link GpuEngine}, which drives the
 * existing generic-index {@code solvePBP}/{@code solveRepairMode} kernels for the live production
 * pipeline. That class's Blackwood-bias machinery (candidate-order jitter tables biasing a
 * different, generic index) has no role here; mixing this engine's own memory layout and launch
 * cadence into it would add risk to the live pipeline for no reuse benefit.
 *
 * <p>Unlike {@link GpuEngine} (which uploads its constant tables once, in the constructor, since
 * they never change for the life of the engine), this engine's candidate tables are re-uploaded
 * via {@link #uploadTables} once per EPOCH (many {@link #runBlackwoodDfs} calls share one table
 * generation -- see {@code BlackwoodGpuRunner.EPOCH_LAUNCHES}), not once per launch.</p>
 *
 * <p>2026-08-04: each thread's in-progress search state now persists across launches in global
 * memory (the {@code d_persist*} buffers below) instead of being discarded every launch -- a
 * thread's search genuinely continues where it left off. {@link #resetEpoch()} forces every
 * thread back to a fresh attempt, which must happen whenever {@link #uploadTables} rebuilds the
 * candidate tables (a persisted resume cursor into a now-replaced table would be pointing at the
 * wrong candidates otherwise).</p>
 */
public class BlackwoodGpuEngine {

    private static final int MAX_THREADS = 20_000;
    // Headroom over the real, measured sizes (BwGpuTablesTest: payload ~38,675, bottomRawPayload
    // exactly 56) -- not the plan's rough pre-implementation estimate (~32,600). Keep these two
    // constants in sync with BwGpuTablesTest's own thresholds if either ever needs to move.
    private static final int MAX_PAYLOAD_SIZE = 50_000;
    private static final int MAX_BOTTOM_PAYLOAD_SIZE = 96;

    private static final String PRODUCTION_PTX = "SolveBlackwoodKernel.ptx";
    private static final String PROFILE_PTX = "SolveBlackwoodKernel.profile.ptx";
    /** Must match BW_PC_SLOTS in SolveBlackwoodKernel.cu. */
    public static final int PROFILE_COUNTER_SLOTS = 16;

    private CUfunction blackwoodDfsFunction;
    private CUfunction blackwoodDfsSharedFunction;
    private CUmodule cuModule;

    /**
     * When true this engine loaded {@link #PROFILE_PTX} (built by
     * build-blackwood-profile-ptx.ps1 with -DBW_PROFILE_COUNTERS), whose kernel takes one extra
     * trailing parameter for the warp-divergence counters. The production PTX has no such
     * parameter, so this flag decides the argument list -- passing the wrong one is an immediate
     * launch failure, not a silent misread.
     */
    private final boolean profilingEnabled;

    // Persistent device buffers -- allocated once, reused every launch.
    private CUdeviceptr d_payload;
    private CUdeviceptr d_gpuHighScore;
    private CUdeviceptr d_bestBoardOut;
    private CUdeviceptr d_solution;
    private CUdeviceptr d_solvedFlag;
    private CUdeviceptr d_totalNodes;
    private CUdeviceptr d_threadDepths;

    // Persistent per-thread search state -- survives across launches within one epoch. See the
    // 2026-08-04 class-level note and SolveBlackwoodKernel.cu's own header comment.
    private CUdeviceptr d_persistBoard;
    private CUdeviceptr d_persistPieceIndexToTryNext;
    private CUdeviceptr d_persistCumulativeBreaks;
    private CUdeviceptr d_persistCumulativeHeuristicSideCount;
    private CUdeviceptr d_persistPieceUsedBits;
    private CUdeviceptr d_persistBsOffset;
    private CUdeviceptr d_persistBsCount;
    private CUdeviceptr d_persistBsPayload;
    private CUdeviceptr d_persistRngState;
    private CUdeviceptr d_persistSolveIndex;
    private CUdeviceptr d_persistBestBoard;
    private CUdeviceptr d_persistBestPiecesPlaced;
    private CUdeviceptr d_needsInit;
    private CUdeviceptr d_profileCounters; // null unless profilingEnabled

    // Seeding from previously saved deep boards. numSeeds == 0 keeps the original
    // always-start-from-a-random-corner behaviour, so seeding is strictly opt-in.
    private static final int MAX_SEEDS = 512;
    private CUdeviceptr d_seedBoards;
    private CUdeviceptr d_seedDepths;
    private CUdeviceptr d_seedShortfalls;
    private volatile int numSeeds = 0;
    private volatile int maxRetreat = 0;
    private volatile int freshFractionPercent = 0;

    // 2026-08-18: opt-in shared-memory caching of the four hot per-step tables -- see
    // SolveBlackwoodKernel.cu's own header note. Default false preserves existing behaviour
    // (solvePBP's sibling GpuEngine.lookaheadEnabled is the precedent for this convention).
    private volatile boolean sharedCacheEnabled = false;

    public BlackwoodGpuEngine() {
        this(false);
    }

    /**
     * @param profilingEnabled load the instrumented kernel instead of the production one. Only
     *                         {@code BlackwoodGpuProfileHarness} should pass true -- the counters
     *                         add atomic traffic, so a profiling run's throughput is not
     *                         representative of production throughput.
     */
    public BlackwoodGpuEngine(boolean profilingEnabled) {
        this.profilingEnabled = profilingEnabled;
        JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync);
        initCUDA();
    }

    private void initCUDA() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext cuContext = new CUcontext();
        cuCtxCreate(cuContext, 0, device);

        cuModule = new CUmodule();
        cuModuleLoad(cuModule, profilingEnabled ? PROFILE_PTX : PRODUCTION_PTX);

        blackwoodDfsFunction = new CUfunction();
        cuModuleGetFunction(blackwoodDfsFunction, cuModule, "solveBlackwoodDfs");

        blackwoodDfsSharedFunction = new CUfunction();
        cuModuleGetFunction(blackwoodDfsSharedFunction, cuModule, "solveBlackwoodDfsShared");

        allocatePersistentBuffers();
        resetEpoch(); // every thread starts needing a fresh attempt on the very first launch
    }

    private void allocatePersistentBuffers() {
        d_payload      = alloc((long) MAX_PAYLOAD_SIZE * Sizeof.INT);
        d_gpuHighScore = alloc(Sizeof.INT);
        d_bestBoardOut = alloc(256L * Sizeof.INT);
        d_solution     = alloc(256L * Sizeof.INT);
        d_solvedFlag   = alloc(Sizeof.INT);
        d_totalNodes   = alloc(Sizeof.LONG);
        d_threadDepths = alloc((long) MAX_THREADS * Sizeof.INT);

        // Per-thread persistent search state, sized at MAX_THREADS regardless of the actual
        // numThreads a given run uses -- the kernel's own tid >= numThreads guard means any
        // unused tail entries are simply never touched, same convention as d_threadDepths above.
        d_persistBoard                         = alloc((long) MAX_THREADS * 256 * Sizeof.INT);
        d_persistPieceIndexToTryNext            = alloc((long) MAX_THREADS * 256 * Sizeof.INT);
        d_persistCumulativeBreaks               = alloc((long) MAX_THREADS * 256 * Sizeof.INT);
        d_persistCumulativeHeuristicSideCount   = alloc((long) MAX_THREADS * 256 * Sizeof.INT);
        d_persistPieceUsedBits                  = alloc((long) MAX_THREADS * 8 * Sizeof.INT);
        d_persistBsOffset                       = alloc((long) MAX_THREADS * 23 * Sizeof.INT);
        d_persistBsCount                        = alloc((long) MAX_THREADS * 23 * Sizeof.INT);
        d_persistBsPayload                      = alloc((long) MAX_THREADS * MAX_BOTTOM_PAYLOAD_SIZE * Sizeof.INT);
        d_persistRngState                       = alloc((long) MAX_THREADS * Sizeof.LONG);
        d_persistSolveIndex                     = alloc((long) MAX_THREADS * Sizeof.INT);
        d_persistBestBoard                      = alloc((long) MAX_THREADS * 256 * Sizeof.INT);
        d_persistBestPiecesPlaced               = alloc((long) MAX_THREADS * Sizeof.INT);
        d_needsInit                             = alloc((long) MAX_THREADS * Sizeof.INT);

        d_seedBoards     = alloc((long) MAX_SEEDS * 256 * Sizeof.INT);
        d_seedDepths     = alloc((long) MAX_SEEDS * Sizeof.INT);
        d_seedShortfalls = alloc(Sizeof.INT);

        if (profilingEnabled) {
            d_profileCounters = alloc((long) PROFILE_COUNTER_SLOTS * Sizeof.LONG);
            resetProfileCounters();
        }
    }

    /**
     * Supplies saved boards for threads to resume from instead of starting at a random corner.
     * Each seed is a 256-entry step-ordered array of {@code (pieceNumber << 2) | rotation}, negative
     * where the seed ends -- see {@code BwSeedLoader}.
     *
     * <p>{@code maxRetreat} is how far back from a seed's own tip a thread may randomly pull before
     * resuming. Some spread is essential: candidate ORDER is global, so threads resuming the same
     * board at the same depth would walk identical orders and duplicate each other's work.</p>
     *
     * <p>Takes effect on the next fresh attempt (i.e. after {@link #resetEpoch()}); already-running
     * threads keep the search state they have.</p>
     *
     * @param seeds      step-ordered encodings, each 256 long
     * @param depths     how many steps each seed covers
     * @param maxRetreat 0 means every thread resumes at its seed's full depth
     * @param freshFractionPercent percentage of attempts that ignore the seeds entirely and start
     *                             from a random corner. 0 means every attempt seeds, which confines
     *                             the whole population to variations of the supplied boards -- with
     *                             no way for a genuinely new board to enter the pool.
     */
    public void uploadSeeds(List<int[]> seeds, int[] depths, int maxRetreat, int freshFractionPercent) {
        if (seeds.size() != depths.length) {
            throw new IllegalArgumentException("seeds/depths length mismatch: " + seeds.size() + " vs " + depths.length);
        }
        if (seeds.size() > MAX_SEEDS) {
            throw new IllegalArgumentException("seed count " + seeds.size() + " exceeds MAX_SEEDS=" + MAX_SEEDS);
        }
        if (maxRetreat < 0) throw new IllegalArgumentException("maxRetreat must be >= 0, was " + maxRetreat);
        if (freshFractionPercent < 0 || freshFractionPercent > 100) {
            throw new IllegalArgumentException("freshFractionPercent must be 0..100, was " + freshFractionPercent);
        }

        if (seeds.isEmpty()) {
            this.numSeeds = 0;
            return;
        }

        int[] flat = new int[seeds.size() * 256];
        for (int s = 0; s < seeds.size(); s++) {
            int[] seed = seeds.get(s);
            if (seed.length != 256) {
                throw new IllegalArgumentException("seed " + s + " has length " + seed.length + ", expected 256");
            }
            System.arraycopy(seed, 0, flat, s * 256, 256);
        }
        cuMemcpyHtoD(d_seedBoards, Pointer.to(flat), (long) flat.length * Sizeof.INT);
        cuMemcpyHtoD(d_seedDepths, Pointer.to(depths), (long) depths.length * Sizeof.INT);
        this.numSeeds = seeds.size();
        this.maxRetreat = maxRetreat;
        this.freshFractionPercent = freshFractionPercent;
    }

    /** Number of seeds currently in use; 0 means threads start from a random corner. */
    public int getNumSeeds() {
        return numSeeds;
    }

    /**
     * When true, {@link #runBlackwoodDfs} launches {@code solveBlackwoodDfsShared} instead of
     * {@code solveBlackwoodDfs} -- identical search logic, but the four hot per-step tables
     * (stepToTableId/stepBoardIdx/breakArray/heuristicArray, read on every outer-loop iteration)
     * are cached in block-local {@code __shared__} memory instead of {@code __constant__} memory.
     * Default false preserves existing behaviour until deliberately flipped.
     */
    public boolean isSharedCacheEnabled() {
        return sharedCacheEnabled;
    }

    public void setSharedCacheEnabled(boolean enabled) {
        this.sharedCacheEnabled = enabled;
    }

    /**
     * Reads back every thread's own best-ever board this epoch ({@code d_persistBestBoard}), as
     * {@code numThreads} consecutive 256-entry blocks of packed candidate records.
     *
     * <p>{@link #runBlackwoodDfs} only ever surfaces the single globally deepest board, which is
     * not enough to judge a population: with seeding on, that one board is usually just the deepest
     * seed replayed back, identical across configurations. Reading the whole population is what
     * makes it possible to ask how many DISTINCT boards a configuration actually produced and how
     * good they are -- the questions that decide whether exploration is paying for itself.</p>
     */
    public int[] readThreadBestBoards(int numThreads) {
        if (numThreads > MAX_THREADS) {
            throw new IllegalArgumentException("numThreads " + numThreads + " exceeds MAX_THREADS=" + MAX_THREADS);
        }
        int[] out = new int[numThreads * 256];
        cuMemcpyDtoH(Pointer.to(out), d_persistBestBoard, (long) numThreads * 256 * Sizeof.INT);
        return out;
    }

    /**
     * Threads whose seed replay stopped short of its target depth since the last reset. Persistently
     * high means the seed boards are not reachable through the current candidate tables -- e.g.
     * boards produced under a different piece numbering or a different break schedule.
     */
    public int readAndResetSeedShortfalls() {
        int[] out = new int[1];
        cuMemcpyDtoH(Pointer.to(out), d_seedShortfalls, Sizeof.INT);
        cuMemcpyHtoD(d_seedShortfalls, Pointer.to(new int[]{0}), Sizeof.INT);
        return out[0];
    }

    /** Zeroes the divergence counters. Call before a measured batch of launches. */
    public void resetProfileCounters() {
        if (!profilingEnabled) throw new IllegalStateException("engine was not built with profiling enabled");
        cuMemcpyHtoD(d_profileCounters, Pointer.to(new long[PROFILE_COUNTER_SLOTS]),
                (long) PROFILE_COUNTER_SLOTS * Sizeof.LONG);
    }

    /** Reads the raw counter slots; indices match the BW_PC_* defines in SolveBlackwoodKernel.cu. */
    public long[] readProfileCounters() {
        if (!profilingEnabled) throw new IllegalStateException("engine was not built with profiling enabled");
        long[] out = new long[PROFILE_COUNTER_SLOTS];
        cuMemcpyDtoH(Pointer.to(out), d_profileCounters, (long) PROFILE_COUNTER_SLOTS * Sizeof.LONG);
        return out;
    }

    /**
     * Forces every thread to start a fresh attempt on its next launch, discarding any persisted
     * in-progress state. Must be called whenever {@link #uploadTables} replaces the candidate
     * tables -- a persisted resume cursor into a now-stale table would otherwise resume into the
     * wrong candidates. Also called once from the constructor for the very first launch.
     */
    public void resetEpoch() {
        int[] ones = new int[MAX_THREADS];
        Arrays.fill(ones, 1);
        cuMemcpyHtoD(d_needsInit, Pointer.to(ones), (long) MAX_THREADS * Sizeof.INT);
    }

    private static CUdeviceptr alloc(long bytes) {
        CUdeviceptr p = new CUdeviceptr();
        cuMemAlloc(p, bytes);
        return p;
    }

    private void uploadConstant(String symbol, int[] data, long bytes) {
        CUdeviceptr ptr = new CUdeviceptr();
        long[] size = new long[1];
        cuModuleGetGlobal(ptr, size, cuModule, symbol);
        cuMemcpyHtoD(ptr, Pointer.to(data), bytes);
    }

    /**
     * Uploads a fresh candidate-table set -- call once per batch, before {@link #runBlackwoodDfs}.
     * Unlike {@link GpuEngine}'s one-time constant uploads, this is meant to be called repeatedly
     * (once per {@code BlackwoodSolver.prepare()} + {@code BwGpuTables.build()} cycle).
     */
    public void uploadTables(BwGpuTables.GpuTableSet tables) {
        if (tables.payload().length > MAX_PAYLOAD_SIZE) {
            throw new IllegalStateException("candidate payload size " + tables.payload().length
                    + " exceeds MAX_PAYLOAD_SIZE=" + MAX_PAYLOAD_SIZE + " -- bump the buffer (see BwGpuTablesTest for the real measured size)");
        }
        if (tables.bottomRawPayload().length > MAX_BOTTOM_PAYLOAD_SIZE) {
            throw new IllegalStateException("bottomRawPayload size " + tables.bottomRawPayload().length
                    + " exceeds MAX_BOTTOM_PAYLOAD_SIZE=" + MAX_BOTTOM_PAYLOAD_SIZE + " -- bump the buffer");
        }

        long csrBytes = (long) BwGpuTables.NUM_TABLES * BwGpuTables.KEY_SPACE * Sizeof.INT;
        uploadConstant("c_csrOffset", tables.csrOffset(), csrBytes);
        uploadConstant("c_csrCount", tables.csrCount(), csrBytes);
        uploadConstant("c_bottomRawOffset", tables.bottomRawOffset(), 23L * Sizeof.INT);
        uploadConstant("c_bottomRawCount", tables.bottomRawCount(), 23L * Sizeof.INT);

        // c_bottomRawPayload is a fixed-size __constant__ array (MAX_BOTTOM_PAYLOAD=96 in the
        // .cu) -- pad the real, shorter payload with zeros rather than uploading a mismatched byte count.
        int[] paddedBottomPayload = Arrays.copyOf(tables.bottomRawPayload(), MAX_BOTTOM_PAYLOAD_SIZE);
        uploadConstant("c_bottomRawPayload", paddedBottomPayload, (long) MAX_BOTTOM_PAYLOAD_SIZE * Sizeof.INT);

        uploadConstant("c_stepToTableId", tables.stepToTableId(), 256L * Sizeof.INT);
        uploadConstant("c_stepBoardIdx", tables.stepBoardIdx(), 256L * Sizeof.INT);
        uploadConstant("c_breakArray", tables.breakArray(), 256L * Sizeof.INT);
        uploadConstant("c_heuristicArray", tables.heuristicArray(), 256L * Sizeof.INT);

        cuMemcpyHtoD(d_payload, Pointer.to(tables.payload()), (long) tables.payload().length * Sizeof.INT);
    }

    public record GpuResult(int newHighScore, boolean solved, long nodesTaken, int[] threadDepths) {
    }

    /**
     * Launches one batch: {@code numThreads} threads, each running one full Blackwood attempt
     * from a fresh step-0 seed (no CPU-supplied partial boards, unlike {@link GpuEngine#runDeepDfs}
     * -- Blackwood's algorithm always starts fresh). {@code bestBoardOut} is only overwritten if
     * this launch found a new high score or a genuine full solve -- same out-parameter convention
     * as {@code GpuEngine.runDeepDfs}, so check {@code newHighScore > currentHighScore} or
     * {@code solved} before trusting its contents.
     */
    public GpuResult runBlackwoodDfs(long seedBase, long stepBudget, int numThreads,
                                      int currentHighScore, int[] bestBoardOut) {
        if (numThreads > MAX_THREADS) {
            throw new IllegalArgumentException("numThreads " + numThreads + " exceeds MAX_THREADS=" + MAX_THREADS);
        }

        cuMemcpyHtoD(d_gpuHighScore, Pointer.to(new int[]{currentHighScore}), Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag, Pointer.to(new int[]{0}), Sizeof.INT);
        cuMemcpyHtoD(d_totalNodes, Pointer.to(new long[]{0L}), Sizeof.LONG);
        cuMemcpyHtoD(d_threadDepths, Pointer.to(new int[numThreads]), (long) numThreads * Sizeof.INT);

        Pointer[] params = {
                Pointer.to(d_payload),
                Pointer.to(new long[]{seedBase}),
                Pointer.to(new long[]{stepBudget}),
                Pointer.to(new int[]{numThreads}),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_totalNodes),
                Pointer.to(d_threadDepths),
                Pointer.to(d_persistBoard),
                Pointer.to(d_persistPieceIndexToTryNext),
                Pointer.to(d_persistCumulativeBreaks),
                Pointer.to(d_persistCumulativeHeuristicSideCount),
                Pointer.to(d_persistPieceUsedBits),
                Pointer.to(d_persistBsOffset),
                Pointer.to(d_persistBsCount),
                Pointer.to(d_persistBsPayload),
                Pointer.to(d_persistRngState),
                Pointer.to(d_persistSolveIndex),
                Pointer.to(d_persistBestBoard),
                Pointer.to(d_persistBestPiecesPlaced),
                Pointer.to(d_needsInit),
                Pointer.to(d_seedBoards),
                Pointer.to(d_seedDepths),
                Pointer.to(new int[]{numSeeds}),
                Pointer.to(new int[]{maxRetreat}),
                Pointer.to(new int[]{freshFractionPercent}),
                Pointer.to(d_seedShortfalls)
        };
        if (profilingEnabled) {
            // The instrumented kernel's one extra trailing parameter -- see the .cu's #ifdef block.
            params = Arrays.copyOf(params, params.length + 1);
            params[params.length - 1] = Pointer.to(d_profileCounters);
        }
        Pointer kernelParameters = Pointer.to(params);

        int blockSize = 256;
        int gridSize = (int) Math.ceil((double) numThreads / blockSize);
        CUfunction fn = sharedCacheEnabled ? blackwoodDfsSharedFunction : blackwoodDfsFunction;
        cuLaunchKernel(fn, gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        int[] resultHighScore = new int[1];
        long[] totalNodes = new long[1];
        int[] solved = new int[1];
        int[] threadDepths = new int[numThreads];

        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(totalNodes), d_totalNodes, Sizeof.LONG);
        cuMemcpyDtoH(Pointer.to(solved), d_solvedFlag, Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(threadDepths), d_threadDepths, (long) numThreads * Sizeof.INT);

        if (resultHighScore[0] > currentHighScore) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        }
        if (solved[0] == 1) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution, 256L * Sizeof.INT);
        }

        return new GpuResult(resultHighScore[0], solved[0] == 1, totalNodes[0], threadDepths);
    }
}
