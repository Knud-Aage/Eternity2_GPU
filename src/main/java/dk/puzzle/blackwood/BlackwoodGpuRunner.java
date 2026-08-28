package dk.puzzle.blackwood;

import dk.puzzle.core.PieceLoader;
import dk.puzzle.io.drive.DriveUploader;
import dk.puzzle.gpu.BlackwoodGpuEngine;
import dk.puzzle.model.PieceInventory;
import dk.puzzle.tools.HoleSolver;
import dk.puzzle.util.PieceUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Standalone launcher for the GPU-native Blackwood kernel ({@code SolveBlackwoodKernel.cu}),
 * mirroring {@link BlackwoodSolver}'s own standalone-launcher convention (own {@code main()}, no
 * GUI/{@code StartupDialog} wiring -- see {@code run-blackwood.cmd}).
 *
 * <p>Each GPU launch resumes every thread's persisted in-progress search rather than restarting it
 * (see {@code SolveBlackwoodKernel.cu}'s 2026-08-04 header note) -- so table rebuilds can no longer
 * happen every launch (a resumed cursor into a replaced table would point at the wrong candidates).
 * Instead, {@link BlackwoodSolver#prepare()} + {@link BwGpuTables#build} + a full thread-state reset
 * ({@link BlackwoodGpuEngine#resetEpoch()}) happen only once every {@link #EPOCH_LAUNCHES} launches
 * ("epoch" boundaries); every other launch just resumes. {@code solver} here is used only for its
 * {@code prepare()} output (the candidate tables) -- {@code solvePuzzle()}/{@code run()} are never
 * called; the actual search happens entirely on the GPU.</p>
 */
public class BlackwoodGpuRunner {

    private static final Logger logger = LogManager.getLogger(BlackwoodGpuRunner.class);

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    // Dedicated, small, append-only file for just the completed links -- eternity_solver.log is
    // shared with every other Java tool in this project and fills up fast (7.9MB / 36k GPU lines
    // alone as of 2026-08-17), so finding a link there means scrolling past everything else. A
    // direct file write here rather than a second log4j appender, since log4j2.xml is shared config
    // touching every logger in the project -- not worth the risk of misrouting something else's
    // output to get one dedicated file right.
    private static final Path COMPLETED_LINKS_LOG = Path.of("logs", "gpu_completed_links.log");
    private static final int SAVE_THRESHOLD = 190; // matches BlackwoodSolver's own default, for direct comparability
    // 2026-08-17, measured (BlackwoodGpuBreadthDepthHarness): 16384 was chosen for SM saturation --
    // i.e. for raw node throughput -- but throughput turns out to be nearly irrelevant to the metric
    // that matters. At equal wall-clock, 16384 threads searched 47.5 BILLION nodes and 64 threads
    // searched 0.55 billion (86x fewer), yet reached the same max depth within 2 pieces. Spreading
    // the budget over more lineages just makes every lineage shallower: at equal TOTAL NODES, max
    // depth went 160 (16384 threads) -> 246 (64 threads).
    // 1024 keeps the population mean depth high without giving up the parallelism entirely; it and
    // 16384 tied on max depth at equal wall-clock, but 1024 held a much deeper mean (205 vs 174).
    // 2026-08-23: that comparison never actually included 64 or 256 -- BlackwoodGpuBreadthDepthHarness
    // later found maxDepth flat (243-244) across 64..16384 threads at equal wall-clock, with meanDepth
    // *favouring* fewer threads (151.6 at 64 vs 137.9 at 1024). A same-day quality harness scoring
    // actual post-HoleSolver conflicts found no comparably large effect, but on too small a sample
    // (18-56 boards) to rule one out. Overridable so a real overnight run can test it properly.
    private static final int NUM_THREADS = parseIntEnv("ETERNITY_GPU_NUM_THREADS", 1024);
    // 2026-08-18, measured (BlackwoodGpuTdrCeilingHarness): a single launch survived cleanly up to
    // 171,261 ms (stepBudget=25,600,000) with no CUDA/TDR failure -- the "~2000ms WDDM TDR default"
    // this band used to target was never real on this machine/driver, off by roughly 5000x. The two
    // "long launch" log outliers that looked like TDR survival earlier (751,813 ms and 9,857,717 ms)
    // turned out to be sleep/resume artifacts instead (confirmed against Windows Event Log 42/107),
    // not evidence either way.
    //
    // Retargeted to settle around several-second launches -- ~15-40x inside the proven-safe margin,
    // not the measured ceiling itself, since a longer single launch also delays when a new record
    // gets detected/logged/saved (nothing is checked until the launch returns). The payoff mirrors
    // EPOCH_LAUNCHES' own 2026-08-17 finding just above: a longer launch stretches how much wall-clock
    // time elapses before the same EPOCH_LAUNCHES=20,000 launch count forces every thread's persisted
    // search back to a fresh start, extending sustained per-lineage backtracking the same way removing
    // the old 60-launch epoch reset did.
    private static final long INITIAL_STEP_BUDGET = 1_000_000L;
    private static final long MIN_STEP_BUDGET = 1_000L;
    private static final long FAST_LAUNCH_MILLIS = 4_000L;  // below this, double the budget next launch
    private static final long SLOW_LAUNCH_MILLIS = 10_000L; // above this, halve it

    // How many launches share one table generation + persisted thread-search-state before
    // everything resets fresh (a table rebuild re-randomizes candidate order, so every resume
    // cursor into the old tables must be discarded with it -- see BlackwoodGpuEngine.resetEpoch).
    //
    // 2026-08-17, measured (BlackwoodGpuEpochResetHarness), equal wall-clock per arm, reset being
    // the ONLY difference:
    //     16384 threads: mean population depth  76.7 (reset every 60)  ->  174.4 (never)
    //      1024 threads: mean population depth 100.6 (reset every 60)  ->  205.1 (never)
    // Max depth likewise 236->248 and 245->248. The old value was costing roughly HALF the
    // population's accumulated depth: climbing from scratch to ~247 takes ~30s, and at this
    // launch cadence a 60-launch epoch is only about a minute, so most of every epoch was spent
    // re-covering ground the previous epoch had already covered, over and over.
    //
    // The rebuild's purpose is diversification, but that is already supplied per-thread (each
    // thread re-randomizes its own bottomSides every attempt from its own RNG stream), so the
    // global rebuild was buying very little of it at a very high price. Kept rather than removed
    // outright so tables never go stale indefinitely -- just at an interval that no longer
    // truncates the sustained backtracking this algorithm depends on.
    private static final long EPOCH_LAUNCHES = 20_000;

    // Seeding from previously saved deep boards. The kernel reaches ~247 pieces from scratch in
    // about 30 seconds but took three days of running to reach 251, so nearly all of the compute
    // spent re-deriving the first ~247 pieces is spent re-covering known ground. Resuming from the
    // deepest boards already on disk puts every thread at the frontier instead.
    // MIN_SEED_DEPTH is deliberately just below the best boards on disk -- shallower boards would
    // dilute the pool without adding frontier coverage.
    private static final int MIN_SEED_DEPTH = 245;
    private static final int MAX_SEEDS = 256;
    // How far back from a seed's tip a thread may randomly pull before resuming. Needed for
    // diversity (candidate order is global, so same board + same depth = duplicated work), and it
    // also lets threads explore alternatives that branch off well below the tip.
    //
    // 2026-08-17, A/B'd (BlackwoodGpuRetreatHarness), 4 arms x 180s, fresh held at 10% so retreat
    // is the only variable. Novel = distinct population best-boards that are not a seed handed back
    // unchanged; conflicts are real HoleSolver completions over a sample of those:
    //     retreat   novel   bestConf   medConf
    //          40      98         13        14
    //         100     122         12        14
    //         180     215         13        19
    //         250     301         13        20
    // 100 dominates the old 40 on all three: more novel boards, better best, equal median. And it
    // was the only arm to produce a NOVEL 12-conflict board -- one not in the seed pool -- i.e. the
    // first genuinely independent record-tying board the GPU has produced rather than replayed.
    //
    // Past ~100 the same trade the fresh fraction showed reappears: much more diversity, much worse
    // quality. Unfreezing most of the board discards the very structure that made the seed good.
    //
    // Caveat: bestConf is a min over a 25-board sample, so the single 12 could be luck. What
    // actually justifies 100 over 40 is that its MEDIAN is equal-best while producing 24% more
    // novel boards -- the tail result is a bonus, not the argument.
    private static final int MAX_RETREAT = 100;
    // Percentage of attempts that ignore the seeds and start from a random corner.
    //
    // Without this the run is a CLOSED loop: every thread resumes one of a small set of archive
    // boards and only re-explores its last MAX_RETREAT steps, so the population never leaves the
    // neighbourhood of boards the C# solver already searched for days -- plausibly exhausted
    // ground, with no mechanism to look elsewhere. The fresh fraction is what lets a genuinely new
    // board be found, scored, saved, and then picked up as a seed at the next epoch, closing the
    // explore/exploit loop instead of just exploiting.
    //
    // 2026-08-17, A/B'd (BlackwoodGpuFreshFractionHarness), 4 arms x 180s, 1024 threads, identical
    // 59-board seed pool. Novel boards = distinct population best-boards that are NOT a seed handed
    // back unchanged; conflicts are real HoleSolver completions over a sample of those:
    //     fresh%   novel   bestConf   medConf
    //          0      44         13        14
    //         25     203         13        17
    //         50     326         13        19
    //        100     684         18        20
    // The fresh fraction buys diversity and pays for it in quality: 0/25/50 all TIE on best board
    // found, while median quality degrades monotonically, and pure exploration cannot even reach
    // the seeded arms' depth. No arm beat the seed pool's own best (12).
    //
    // Two things this corrects. First, an earlier assumption that 0% is a closed loop that can
    // never produce anything new -- it produced 44 novel boards, the best-quality set of any arm,
    // because retreating up to MAX_RETREAT steps and re-searching IS exploration, just local to
    // known-good boards. Second, the original 25% guess: it is strictly worse than 0% here.
    //
    // Kept small but non-zero rather than 0 because the experiment measures the quality
    // DISTRIBUTION over 3 minutes, not the rare long-horizon event exploration actually exists for
    // (escaping a basin the whole seed pool may share). That payoff is not measurable at this
    // timescale, so this is a deliberate hedge, not a measured optimum.
    private static final int FRESH_FRACTION_PERCENT = 10;
    // Candidates to score before ranking. Scoring runs HoleSolver once per board (~1s each), and
    // only at an epoch boundary, so this bounds a startup cost rather than a per-launch one.
    private static final int MAX_SEED_CANDIDATES = 120;

    // 2026-08-18: A/B lever, off by default (seeding stays on, matching production). Set
    // ETERNITY_GPU_SEEDING=false to test pure random-restart search -- WITH the epoch-reset and
    // thread-count fixes still active, unlike the original unseeded runs that motivated seeding in
    // the first place. Exists because the evidence since has pointed the other way: seeded search's
    // own duplicate-detection sweep found 28 unique boards out of 59 saved (more than half were the
    // same board re-derived from a different retreat point), and both the retreat and fresh-fraction
    // A/Bs showed perturbing away from the seed makes results MORE diverse and WORSE, never better --
    // the signature of a narrow local optimum, not a neighbourhood with better boards nearby. Matches
    // Blackwood's own account of his 470: a month of continuous pure random-restart search on one
    // PC, his own word for it "luck" -- not refinement of a near-miss.
    private static final boolean SEEDING_ENABLED =
            !"false".equalsIgnoreCase(System.getenv("ETERNITY_GPU_SEEDING"));

    // 2026-08-18, verified (BlackwoodGpuSharedCacheHarness): bit-identical results to the
    // constant-memory kernel across chained launches (same highScore, nodesTaken, threadDepths,
    // best board), so this is a pure speed change, not a behaviour change. Equal-GPU-time A/B
    // (180s/arm): 206 launches vs 153 (+35%), max depth 247 vs 245, mean depth 226.4 vs 223.7.
    // Divergence profile unaffected either way (100% warp efficiency both configurations, as
    // expected -- shared memory targets __constant__ cache-miss latency, not warp lockstep).
    private static final boolean SHARED_CACHE_ENABLED =
            !"false".equalsIgnoreCase(System.getenv("ETERNITY_GPU_SHARED_CACHE"));

    private static int parseIntEnv(String name, int defaultValue) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) return defaultValue;
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            System.err.println("Ignoring invalid " + name + "=" + v + ", using default " + defaultValue);
            return defaultValue;
        }
    }

    // Trials for HoleSolver's completion pass when scoring a candidate save (see trySave).
    // Matches the C# solver's own established choice (Util.cs's TryLabelWithConflictCount) rather
    // than HoleSolver's much heavier 200,000-trial CLI default -- this runs in the same thread as
    // the main launch loop, only on the (already rare) event of a new per-run depth record, but
    // still shouldn't stall launches for longer than that cadence tolerates.
    private static final int SCORING_TRIALS = 5000;

    // Population harvesting.
    //
    // 2026-08-17: without this the run can save NOTHING. trySave is only reached when a launch
    // beats the running depth high score, and that high score sits at 252 (recovered from disk) --
    // so a 250-piece board completing to 11 conflicts is thrown away without ever being scored.
    // Fixing the save criterion to use conflicts (earlier today) fixed the wrong level: the GATE
    // was still depth. Confirmed empirically -- the retreat A/B found a novel 12-conflict board
    // that production would have discarded, because it was not deeper than 252.
    //
    // So periodically read the whole population's best boards, score the deepest previously-unseen
    // ones, and save on conflicts alone. Interval and sample were originally sized so scoring
    // (~1s/board, estimated) would stay a few percent of wall time: 12 boards per ~300 launches
    // as ~12s per ~3.5 minutes.
    //
    // 2026-08-25: raised 12->100. Measured cost is actually ~0.3s/board (eternity_solver.log:
    // 12 boards score in 3-4s wall clock per harvest), and the eligible population dwarfs the old
    // sample -- a launch sampled at the same time showed depth[min=244 mean=246.6 max=251] across
    // all 16384 threads, i.e. the ENTIRE population already clears HARVEST_MIN_DEPTH=240, so a
    // top-12-by-depth cut was scoring under 0.1% of it. That matters because depth doesn't cleanly
    // predict the post-HoleSolver conflict count (observed directly: depth-250 boards scoring
    // 15/15/17 conflicts alongside depth-248 boards ranging 14-18), so a narrow top-N window misses
    // better boards sitting just below the max depth. 100 boards/harvest is ~30s, still only ~1-2%
    // of the ~22-25min interval between harvests at current launch speed -- HARVEST_INTERVAL has
    // enough headroom on its own that it doesn't need to shrink to afford the bigger sample.
    private static final int HARVEST_INTERVAL = 300;
    private static final int HARVEST_SAMPLE = 100;
    private static final int HARVEST_MIN_DEPTH = 240;
    /** Boards already scored, so a stable population isn't re-scored every harvest. */
    private static final Set<String> harvestedFingerprints = new HashSet<>();
    private static final int HARVEST_MEMORY_CAP = 200_000;
    /**
     * Fingerprints of the current seed boards. A thread that replayed its seed and never improved
     * on it still holds that seed as its best board, so without this the harvest re-saves the seed
     * pool back into the output folder under new names -- inflating it with copies of boards that
     * already exist and diluting the next epoch's seed pool with duplicates.
     */
    private static final Set<String> seedFingerprints = new HashSet<>();
    /**
     * Completed boards already saved, keyed by their bucas encoding.
     *
     * <p>Filtering seed replays by PARTIAL board is not enough: two different partial boards in the
     * same lineage can complete to the identical 256-piece board, so a "novel" partial can still
     * produce a duplicate result. Observed directly -- a harvested board that passed the seed
     * filter completed to a board byte-identical to the existing 12-conflict record. Deduping on
     * the COMPLETED board is the level that actually matters, since that is what gets saved,
     * compared, and reported.</p>
     */
    private static final Set<String> savedCompletedBoards = new HashSet<>();

    public static void main(String[] args) throws Exception {
        // NOT Documents: it is OneDrive-redirected by Known Folder Move on this machine, so every
        // board saved here was being uploaded to the cloud. UserProfile is never redirected.
        // Same reasoning (and the same override-by-env-var escape hatch) as the C# solver's own
        // save path. Override with ETERNITY_GPU_SOLUTIONS_DIR to put boards on another drive.
        String configuredDir = System.getenv("ETERNITY_GPU_SOLUTIONS_DIR");
        Path outputDir = (configuredDir == null || configuredDir.isBlank())
                ? Path.of(System.getProperty("user.home"), "EternitySolutions_GpuBlackwood")
                : Path.of(configuredDir);

        BlackwoodSolver solver = new BlackwoodSolver(SAVE_THRESHOLD, outputDir, 1, PIECES_PATH);

        List<BwPiece> pieces = BwUtil.getPieces(PIECES_PATH);
        BwPiece[] pieceByNumber = new BwPiece[257];
        for (BwPiece p : pieces) {
            pieceByNumber[p.pieceNumber()] = p;
        }
        // Separate from BwGpuTables' Blackwood-raw packing -- this is HoleSolver's own
        // PieceInventory, needed to run its actual completion logic in-process below.
        PieceInventory inventory = new PieceInventory(PieceLoader.loadPieces());

        BlackwoodGpuEngine engine = new BlackwoodGpuEngine();
        engine.setSharedCacheEnabled(SHARED_CACHE_ENABLED);
        long launchCounter = 0;
        int currentHighScore = scanExistingHighScore(outputDir);
        long stepBudget = INITIAL_STEP_BUDGET;

        logger.info("BlackwoodGpuRunner starting. numThreads={}, initialStepBudget={}, saveThreshold={}, epochLaunches={}, resumedHighScore={}, seedingEnabled={}, sharedCacheEnabled={}, breakIndexesAllowed={}",
                NUM_THREADS, INITIAL_STEP_BUDGET, SAVE_THRESHOLD, EPOCH_LAUNCHES, currentHighScore, SEEDING_ENABLED, SHARED_CACHE_ENABLED,
                java.util.Arrays.toString(BwUtil.BREAK_INDEXES_ALLOWED));

        while (true) {
            if (launchCounter % EPOCH_LAUNCHES == 0) {
                solver.prepare();
                BwGpuTables.GpuTableSet tables = BwGpuTables.build(solver);
                engine.uploadTables(tables);
                if (SEEDING_ENABLED) {
                    // Reload seeds at each epoch: boards saved since the last boundary (including
                    // this run's own new records) become seeds for the next one.
                    loadSeeds(engine, tables.stepBoardIdx(), outputDir, inventory);
                } else {
                    engine.uploadSeeds(List.of(), new int[0], 0, 0);
                }
                engine.resetEpoch();
                logger.info("Epoch boundary at launch {}: tables refreshed, {} seed board(s) active, all thread state reset",
                        launchCounter, engine.getNumSeeds());
            }

            long seedBase = System.nanoTime() ^ (launchCounter++ * 0x9E3779B97F4A7C15L);
            int[] bestBoardOut = new int[256];

            long launchStartNanos = System.nanoTime();
            BlackwoodGpuEngine.GpuResult result =
                    engine.runBlackwoodDfs(seedBase, stepBudget, NUM_THREADS, currentHighScore, bestBoardOut);
            long launchMillis = (System.nanoTime() - launchStartNanos) / 1_000_000L;

            // Population depth stats: runBlackwoodDfs has always returned per-thread depths and
            // this loop always discarded them. They are the direct read-out of whether threads are
            // accumulating sustained depth or being repeatedly knocked back to shallow water --
            // exactly the signal that showed the old 60-launch epoch was halving mean depth.
            // Replay shortfalls are the health check on the seed pool: a seed that isn't reachable
            // through the current candidate tables (wrong piece numbering, incompatible break
            // schedule) silently resumes shallower than intended, which would otherwise look like
            // seeding simply not helping.
            int shortfalls = engine.getNumSeeds() > 0 ? engine.readAndResetSeedShortfalls() : 0;
            logger.info("Launch {}: {} threads, stepBudget={}, {} ms, nodesTaken={}, newHighScore={}, solved={}, depth[{}], seedShortfalls={}",
                    launchCounter, NUM_THREADS, stepBudget, launchMillis,
                    result.nodesTaken(), result.newHighScore(), result.solved(),
                    describeDepths(result.threadDepths()), shortfalls);

            if (result.newHighScore() > currentHighScore || result.solved()) {
                currentHighScore = result.newHighScore();
                if (currentHighScore >= SAVE_THRESHOLD) {
                    trySave(bestBoardOut, currentHighScore, pieceByNumber, inventory, outputDir);
                }
            }

            // Depth records are rare once seeded (the pool's own depth is already the ceiling), so
            // this -- not the branch above -- is what actually finds good boards on a seeded run.
            if (launchCounter % HARVEST_INTERVAL == 0) {
                harvestPopulation(engine, result.threadDepths(), pieceByNumber, inventory, outputDir);
            }

            if (launchMillis < FAST_LAUNCH_MILLIS) {
                stepBudget = stepBudget * 2;
            } else if (launchMillis > SLOW_LAUNCH_MILLIS) {
                stepBudget = Math.max(MIN_STEP_BUDGET, stepBudget / 2);
            }
        }
    }

    /**
     * Gathers the deepest boards this project has saved -- from this runner, the CPU Java port, and
     * the C# solver -- and hands them to the GPU as resume points. All three write the same grid
     * format (see {@link BwSeedLoader}). Failure here is never fatal: with no seeds the kernel just
     * starts from random corners as it always did.
     */
    private static void loadSeeds(BlackwoodGpuEngine engine, int[] stepBoardIdx, Path gpuOutputDir,
                                  PieceInventory inventory) {
        try {
            Path home = Path.of(System.getProperty("user.home"));
            List<Path> dirs = List.of(
                    gpuOutputDir,
                    home.resolve("EternitySolutions"),             // C# solver
                    home.resolve("EternitySolutions_drop239"),     // C# solver, tuned break schedule
                    home.resolve("Documents").resolve("EternitySolutions_JavaPort"));

            List<BwSeedLoader.Seed> candidates =
                    BwSeedLoader.load(dirs, MIN_SEED_DEPTH, MAX_SEED_CANDIDATES, stepBoardIdx);
            if (candidates.isEmpty()) {
                logger.info("No seed boards at depth >= {} found; threads will start from random corners", MIN_SEED_DEPTH);
                engine.uploadSeeds(List.of(), new int[0], 0, 0);
                return;
            }

            // Rank by what each board actually completes to, not by depth -- see BwSeedLoader.
            List<BwSeedLoader.Seed> seeds =
                    BwSeedLoader.rankByConflicts(candidates, inventory, SCORING_TRIALS, MAX_SEEDS);

            List<int[]> encoded = new ArrayList<>(seeds.size());
            int[] depths = new int[seeds.size()];
            seedFingerprints.clear();
            for (int i = 0; i < seeds.size(); i++) {
                BwSeedLoader.Seed seed = seeds.get(i);
                encoded.add(seed.stepEncoded());
                depths[i] = seed.depth();
                seedFingerprints.add(fingerprintOfSeed(seed, stepBoardIdx));
            }
            engine.uploadSeeds(encoded, depths, MAX_RETREAT, FRESH_FRACTION_PERCENT);

            BwSeedLoader.Seed best = seeds.get(0);
            BwSeedLoader.Seed worst = seeds.get(seeds.size() - 1);
            logger.info("Seeding from {} of {} candidate board(s): conflicts {}..{}, best is {} pieces -> {} conflicts ({}). maxRetreat={}, freshFraction={}%",
                    seeds.size(), candidates.size(), best.conflicts(), worst.conflicts(),
                    best.depth(), best.conflicts(), best.source().getFileName(),
                    MAX_RETREAT, FRESH_FRACTION_PERCENT);
        } catch (Exception e) {
            logger.warn("Seed loading failed; continuing without seeds", e);
            try {
                engine.uploadSeeds(List.of(), new int[0], 0, 0);
            } catch (Exception ignored) {
                // already unseeded
            }
        }
    }

    /**
     * min/mean/max alone can't answer the question that actually matters mid-run -- "how much of the
     * population is still near the high score, and how much has fallen back?" A mean of 246 is the
     * same whether one thread sits at 250 and the rest at 246, or thousands are at 249-250 and a
     * long tail drags it down. So this also reports how many threads are within 0/1/2/5 of the
     * deepest thread this launch. Costs nothing: threadDepths is already returned every launch.
     */
    private static String describeDepths(int[] depths) {
        if (depths == null || depths.length == 0) return "n/a";
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        long sum = 0;
        for (int d : depths) {
            if (d < min) min = d;
            if (d > max) max = d;
            sum += d;
        }
        int atMax = 0, within1 = 0, within2 = 0, within5 = 0;
        for (int d : depths) {
            if (d == max) atMax++;
            if (d >= max - 1) within1++;
            if (d >= max - 2) within2++;
            if (d >= max - 5) within5++;
        }
        int n = depths.length;
        return String.format("min=%d mean=%.1f max=%d | at max:%d (%.1f%%) within1:%d (%.1f%%) within2:%d (%.1f%%) within5:%d (%.1f%%)",
                min, (double) sum / n, max,
                atMax, 100.0 * atMax / n,
                within1, 100.0 * within1 / n,
                within2, 100.0 * within2 / n,
                within5, 100.0 * within5 / n);
    }

    private static final Pattern LABELLED_NAME = Pattern.compile("^Errors(\\d+)_Base(\\d+)_.*_RawBoard\\.txt$");
    private static final Pattern LEGACY_NAME = Pattern.compile("^(\\d+)_[0-9a-fA-F-]+_\\d+\\.txt$");

    /**
     * Recovers {@code currentHighScore} from the boards already saved on disk from a previous run,
     * instead of always starting bookkeeping at 0. Reads both this method's own {@code ErrorsN_BaseD_...}
     * naming (introduced 2026-08-17 alongside conflict-based save gating -- see {@link #trySave}) and
     * the older {@code "<pieces>_<uuid>_<timestamp>.txt"} files it superseded, so a restart doesn't
     * regress to comparing against 0 just because every save on disk predates the rename.
     */
    static int scanExistingHighScore(Path outputDir) {
        if (!Files.isDirectory(outputDir)) return 0;
        int max = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(outputDir)) {
            for (Path p : stream) {
                String name = p.getFileName().toString();
                Matcher labelled = LABELLED_NAME.matcher(name);
                if (labelled.matches()) {
                    max = Math.max(max, Integer.parseInt(labelled.group(2)));
                    continue;
                }
                Matcher legacy = LEGACY_NAME.matcher(name);
                if (legacy.matches()) {
                    max = Math.max(max, Integer.parseInt(legacy.group(1)));
                }
            }
        } catch (IOException e) {
            logger.warn("Could not scan {} for existing saves, starting currentHighScore from 0", outputDir, e);
            return 0;
        }
        return max;
    }

    /**
     * Depth was the save criterion until 2026-08-17, and it is a poor proxy for what's actually
     * wanted: a board's REAL quality only shows up after hole-filling, and depth doesn't predict it
     * monotonically. Confirmed directly -- a GPU-found 252-piece board completed to 13 conflicts,
     * while a 251-piece board already in the CPU archive completes to 12. Maximizing depth was
     * actively steering the GPU away from its own better boards.
     *
     * <p>Runs the same completion HoleSolver already does for the C# and CPU-port pipelines, in
     * process (no subprocess -- this IS Java, {@link HoleSolver}'s methods are directly callable),
     * and only saves if the REAL conflict count is competitive. This also incidentally fixes a
     * second problem: a board that's just a replayed, unmodified seed cannot beat the board it came
     * from, so it no longer gets saved back under a new name.</p>
     */
    /**
     * Scores the deepest previously-unseen boards in the live population and saves any that are
     * competitive on conflicts. See HARVEST_INTERVAL for why this exists: the depth-record path
     * alone saves nothing once the seed pool's own depth is already the ceiling.
     */
    private static void harvestPopulation(BlackwoodGpuEngine engine, int[] threadDepths,
                                          BwPiece[] pieceByNumber, PieceInventory inventory, Path outputDir) {
        try {
            int numThreads = threadDepths.length;
            int[] allBoards = engine.readThreadBestBoards(numThreads);

            // Deepest first: a deeper board leaves fewer holes for completion to guess at, so it is
            // the better use of a bounded scoring budget.
            Integer[] order = new Integer[numThreads];
            for (int i = 0; i < numThreads; i++) order[i] = i;
            java.util.Arrays.sort(order, (a, b) -> Integer.compare(threadDepths[b], threadDepths[a]));

            if (harvestedFingerprints.size() > HARVEST_MEMORY_CAP) harvestedFingerprints.clear();

            int scored = 0;
            int skippedSeeds = 0;
            // Conflict counts of everything scored this harvest. The per-board lines already carry
            // these, but they're spread across up to HARVEST_SAMPLE lines buried between launch
            // lines -- this rolls them into one number so "what quality is the population actually
            // producing right now" is readable at a glance, without grepping.
            List<Integer> harvestConflicts = new ArrayList<>();
            for (int idx = 0; idx < numThreads && scored < HARVEST_SAMPLE; idx++) {
                int t = order[idx];
                if (threadDepths[t] < HARVEST_MIN_DEPTH) break; // sorted, so nothing deeper remains
                int[] board = java.util.Arrays.copyOfRange(allBoards, t * 256, (t + 1) * 256);
                String fp = fingerprintOf(board);
                if (seedFingerprints.contains(fp)) { skippedSeeds++; continue; } // a seed handed back, not a find
                if (!harvestedFingerprints.add(fp)) continue;
                int conflicts = evaluateAndMaybeSave(board, threadDepths[t], pieceByNumber, inventory, outputDir, false);
                if (conflicts >= 0) harvestConflicts.add(conflicts);
                scored++;
            }
            if (scored > 0 || skippedSeeds > 0) {
                String conflictSummary = "none scored";
                if (!harvestConflicts.isEmpty()) {
                    List<Integer> sorted = new ArrayList<>(harvestConflicts);
                    java.util.Collections.sort(sorted);
                    conflictSummary = String.format("best=%d median=%d worst=%d",
                            sorted.get(0), sorted.get(sorted.size() / 2), sorted.get(sorted.size() - 1));
                }
                logger.info("Harvest: scored {} new board(s), skipped {} seed replay(s), conflicts[{}]",
                        scored, skippedSeeds, conflictSummary);
            }
        } catch (Exception e) {
            logger.warn("Population harvest failed", e);
        }
    }

    /**
     * Same cell-wise identity as {@link #fingerprintOf}, but built from a seed's step-ordered
     * encoding. The two must agree exactly or seed replays won't be recognised in the harvest.
     */
    private static String fingerprintOfSeed(BwSeedLoader.Seed seed, int[] stepBoardIdx) {
        int[] byBoardIdx = new int[256];
        java.util.Arrays.fill(byBoardIdx, -1);
        for (int step = 0; step < seed.depth(); step++) {
            byBoardIdx[stepBoardIdx[step]] = seed.stepEncoded()[step];
        }
        StringBuilder sb = new StringBuilder(1024);
        for (int i = 0; i < 256; i++) {
            if (byBoardIdx[i] < 0) { sb.append("..,"); continue; }
            sb.append(byBoardIdx[i] >> 2).append(':').append(byBoardIdx[i] & 3).append(',');
        }
        return sb.toString();
    }

    /** Cell-wise (pieceNumber,rotation) identity, for skipping boards already scored. */
    private static String fingerprintOf(int[] board) {
        StringBuilder sb = new StringBuilder(1024);
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1) { sb.append("..,"); continue; }
            BwRotatedPiece p = BwGpuTables.unpack(board[i]);
            sb.append(p.pieceNumber()).append(':').append(p.rotations()).append(',');
        }
        return sb.toString();
    }

    static void trySave(int[] board, int maxSolveIndex, BwPiece[] pieceByNumber,
                                PieceInventory inventory, Path outputDir) {
        evaluateAndMaybeSave(board, maxSolveIndex, pieceByNumber, inventory, outputDir, true);
    }

    /**
     * @return the board's post-completion conflict count, whether or not it was saved, or -1 if
     *   scoring failed. Callers that only want the side effect (see trySave) can ignore it;
     *   harvestPopulation uses it to summarise the population's current quality in one line.
     */
    private static int evaluateAndMaybeSave(int[] board, int maxSolveIndex, BwPiece[] pieceByNumber,
                                            PieceInventory inventory, Path outputDir, boolean depthRecord) {
        try {
            BwRotatedPiece[] rotatedBoard = new BwRotatedPiece[256];
            for (int i = 0; i < 256; i++) {
                rotatedBoard[i] = (board[i] == -1) ? BwRotatedPiece.EMPTY : BwGpuTables.unpack(board[i]);
            }
            // buildBoardString's own trailing line is the bucas link -- reused as-is rather than
            // re-deriving it, so this goes through the exact same URL encoding trySave always has.
            String boardString = BwUtil.buildBoardString(rotatedBoard, pieceByNumber);
            String link = boardString.substring(boardString.lastIndexOf("https://"));

            // decodeBoardAuto translates Blackwood's raw 0-22 colour numbering (what the GPU board
            // and this link use) into HoleSolver's own packed representation -- required before any
            // of HoleSolver's board logic (PieceUtils.getNorth/East/South/West etc.) is meaningful.
            int[] decoded = HoleSolver.decodeBoardAuto(link, inventory, false);
            HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(decoded, inventory, false, SCORING_TRIALS);
            int[] completed = result.bestBoard();
            int conflicts = countConflicts(completed);

            int bestOnDisk = bestConflictsOnDisk(outputDir);
            int keepThreshold = (bestOnDisk == Integer.MAX_VALUE) ? Integer.MAX_VALUE : bestOnDisk + 1;
            // 2026-08-28: exact=true/false measures how often HoleSolver's primary MRV
            // completion (RegionSolver) fully clears every region vs. falling back to the
            // weaker MCV-heuristic repair (see HoleSolver.solveConflicts) -- repairedBoard()
            // is non-null exactly when the fallback ran. budgetExhausted distinguishes WHY:
            // true means a region hit its step budget inconclusively (more budget/a rewind
            // could plausibly help); false means the region's search tree was fully exhausted,
            // proving no zero-conflict rearrangement exists (more budget cannot change that).
            // Only budgetExhausted=true cases are evidence for building an igorpejic-style
            // adaptive-rewind tail -- see ConflictSolveResult's own doc for the full reasoning.
            boolean exact = result.repairedBoard() == null;
            boolean budgetExhausted = result.anyRegionBudgetExhausted();
            if (conflicts > keepThreshold) {
                // Harvest rejects most of what it scores by design, so this is the dominant line
                // volume in this log once enabled at info level.
                if (depthRecord) {
                    logger.info("Depth record at {} pieces completed to {} conflicts -- not within 1 of best-on-disk ({}), not saving, exact={}, budgetExhausted={}",
                            maxSolveIndex, conflicts, bestOnDisk, exact, budgetExhausted);
                } else {
                    logger.info("Harvested board at {} pieces completed to {} conflicts (best-on-disk {}), not saving, exact={}, budgetExhausted={}",
                            maxSolveIndex, conflicts, bestOnDisk, exact, budgetExhausted);
                }
                return conflicts;
            }

            // Dedupe on the completed board, not the partial one it came from -- see
            // savedCompletedBoards for why the partial-level seed filter is insufficient. This is
            // also the real, playable link for the COMPLETED board -- distinct from the partial-board
            // `link` computed above, which still has holes and is only an intermediate value here.
            String completedLink = dk.puzzle.io.BucasExporter.exportBoard(completed);
            if (!savedCompletedBoards.add(completedLink)) {
                logger.debug("Completed board at {} conflicts already saved, skipping duplicate", conflicts);
                return conflicts;
            }

            Files.createDirectories(outputDir);
            String timeId = LocalTime.now().format(DateTimeFormatter.ofPattern("HHmmss_SSS"));
            String prefix = "Errors" + conflicts + "_Base" + maxSolveIndex + "_" + timeId;
            HoleSolver.writePhysicalLayoutFile(outputDir.resolve(prefix + "_physical_layout.txt").toString(),
                    inventory, result.finalBoard(), result.repairedBoard());
            HoleSolver.writeRawBoardFile(outputDir.resolve(prefix + "_RawBoard.txt").toString(), inventory, completed);
            // Third sibling, in Blackwood's OWN piece numbering (boardString, already computed above
            // to build the partial-board link) -- unlike the two files above, which use HoleSolver's
            // internal numbering, this is what BwSeedLoader can actually read directly as a future
            // seed. Same rationale as the C# solver's Util.cs baseboard rename.
            Files.writeString(outputDir.resolve(prefix + "_baseboard.txt"), boardString);
            logger.info("SAVED [{}]: {} pieces, {} conflicts -> {}, exact={}, budgetExhausted={}",
                    depthRecord ? "depth-record" : "harvest", maxSolveIndex, conflicts, prefix, exact, budgetExhausted);
            // Same convention as the C# solver's Util.cs, so both logs are grep-able the same way.
            logger.info("COMPLETED_LINK {}_RawBoard.txt: {}", prefix, completedLink);
            appendCompletedLink(prefix, conflicts, maxSolveIndex, completedLink);
            DriveUploader.uploadRecord(prefix, conflicts, completedLink, "GPU");

            pruneAboveThreshold(outputDir, conflicts);
            return conflicts;
        } catch (Exception e) {
            logger.error("Failed to evaluate/save board at solveIndex={}", maxSolveIndex, e);
            return -1;
        }
    }

    /** Kept package-private for the harnesses; the runner itself only needs the two wrappers above. */
    static int harvestedCount() {
        return harvestedFingerprints.size();
    }

    /**
     * Appends one line to {@link #COMPLETED_LINKS_LOG}. Best-effort: a failure here must never
     * affect the save that already succeeded, matching this file's established pattern for every
     * other piece of ancillary bookkeeping (retention, dedup).
     */
    private static void appendCompletedLink(String prefix, int conflicts, int depth, String link) {
        try {
            Files.createDirectories(COMPLETED_LINKS_LOG.getParent());
            String timestamp = java.time.LocalDateTime.now()
                    .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            String line = String.format("%s  conflicts=%-3d depth=%-3d  %s_RawBoard.txt  %s%n",
                    timestamp, conflicts, depth, prefix, link);
            Files.writeString(COMPLETED_LINKS_LOG, line, java.nio.charset.StandardCharsets.UTF_8,
                    java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.warn("Could not append to {}", COMPLETED_LINKS_LOG, e);
        }
    }

    /** Edge conflicts among ALL 256 cells of a fully hole-filled board (empty cells are not expected here). */
    static int countConflicts(int[] board) {
        int conflicts = 0;
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                int i = r * 16 + c;
                if (c < 15 && PieceUtils.getEast(board[i]) != PieceUtils.getWest(board[i + 1])) conflicts++;
                if (r < 15 && PieceUtils.getSouth(board[i]) != PieceUtils.getNorth(board[i + 16])) conflicts++;
            }
        }
        return conflicts;
    }

    static int bestConflictsOnDisk(Path outputDir) {
        int best = Integer.MAX_VALUE;
        if (!Files.isDirectory(outputDir)) return best;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(outputDir, "Errors*_RawBoard.txt")) {
            for (Path p : stream) {
                Matcher m = LABELLED_NAME.matcher(p.getFileName().toString());
                if (m.matches()) best = Math.min(best, Integer.parseInt(m.group(1)));
            }
        } catch (IOException e) {
            logger.warn("Could not scan {} for best-on-disk conflicts", outputDir, e);
        }
        return best;
    }

    /**
     * Keeps only boards within 1 conflict of the best currently on disk, deleting the rest --
     * mirrors the policy already validated for the C# solver's output folder (same rationale: this
     * folder would otherwise accumulate every depth record ever reached, most of them superseded
     * within minutes). Recomputed from disk every time rather than tracked as running state, so a
     * later improvement retroactively cleans out now-stale near-misses too, not just future ones.
     */
    static void pruneAboveThreshold(Path outputDir, int justSavedConflicts) {
        try {
            Map<Path, Integer> conflictsByFile = new HashMap<>();
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(outputDir, "Errors*_RawBoard.txt")) {
                for (Path p : stream) {
                    Matcher m = LABELLED_NAME.matcher(p.getFileName().toString());
                    if (m.matches()) conflictsByFile.put(p, Integer.parseInt(m.group(1)));
                }
            }
            if (conflictsByFile.isEmpty()) return;

            int keepThreshold = conflictsByFile.values().stream().mapToInt(Integer::intValue).min().orElse(0) + 1;
            for (Map.Entry<Path, Integer> entry : conflictsByFile.entrySet()) {
                if (entry.getValue() <= keepThreshold) continue;
                Path rawBoardFile = entry.getKey();
                String name = rawBoardFile.getFileName().toString();
                String prefix = name.substring(0, name.length() - "_RawBoard.txt".length());
                Path layoutFile = rawBoardFile.resolveSibling(prefix + "_physical_layout.txt");
                // Third sibling (see evaluateAndMaybeSave) -- pruned together so a Blackwood-numbered
                // seed can't outlive the labelled pair that established its quality, same as the C#
                // solver's baseboard rename fixes the equivalent orphan problem there.
                Path baseboardFile = rawBoardFile.resolveSibling(prefix + "_baseboard.txt");
                Files.deleteIfExists(rawBoardFile);
                Files.deleteIfExists(layoutFile);
                Files.deleteIfExists(baseboardFile);
            }
        } catch (IOException e) {
            logger.warn("Retention cleanup failed in {}", outputDir, e);
        }
    }
}
