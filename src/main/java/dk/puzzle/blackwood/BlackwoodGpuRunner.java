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
    // Save/retention window is normally "within 1 of best-on-disk", which tightens every time the
    // record improves -- fine near 471, but on the way there it means every future 12 (already a
    // rare result on its own, see README) stops being saved/uploaded/kept the moment anything ever
    // beats it, however far off that might be. This floor keeps <=12 permanently save-worthy no
    // matter how much better bestOnDisk gets, so results stay visible instead of thinning out over
    // time. Applied in both evaluateAndMaybeSave (the save gate) and pruneAboveThreshold (local
    // retention) -- see both. Mirror any change here in BlackwoodSolver.java (this repo and the CPU
    // one) and Util.cs's PruneAboveThreshold.
    private static final int ALWAYS_SAVE_AT_OR_BELOW = parseIntEnv("ETERNITY_SAVE_FLOOR", 12);
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
    private static final int NUM_THREADS_DEFAULT = 1024;
    private static final int NUM_THREADS = parseIntEnv("ETERNITY_GPU_NUM_THREADS", NUM_THREADS_DEFAULT);
    // True only when ETERNITY_GPU_NUM_THREADS came from the environment. A launch that bypasses
    // run-gpu.cmd (e.g. invoking `java -cp ...` directly, as when restarting from an inspected
    // process command line, which shows arguments but never env vars) silently falls back to the
    // conservative NUM_THREADS_DEFAULT instead of the production-tuned 16384 -- see the startup
    // warning below, added after exactly that happened.
    private static final boolean NUM_THREADS_EXPLICIT = isEnvSet("ETERNITY_GPU_NUM_THREADS");
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
    private static final long EPOCH_LAUNCHES = 20_000;
    private static final int MIN_SEED_DEPTH = 245;
    private static final int MAX_SEEDS = 256;
    // How far back from a seed's tip a thread may randomly pull before resuming. Needed for
    // diversity (candidate order is global, so same board + same depth = duplicated work), and it
    // also lets threads explore alternatives that branch off well below the tip.
    private static final int MAX_RETREAT = 100;
    // Percentage of attempts that ignore the seeds and start from a random corner.
    //
    // Raised 10 -> 40 on 2026-08-30. With persistent duplicate suppression in place the repeat rate
    // became measurable for the first time, and it was 80%: over 352 launches, 10 boards cleared the
    // conflict threshold and 8 were boards already found in earlier runs. At 10% fresh, ~90% of
    // 16384 threads were re-mining seed neighbourhoods that the numbers say are largely exhausted.
    // This is an explore/exploit dial, not a correctness one -- the seeded majority still holds the
    // frontier, this just stops nearly all capacity going to ground that repeats itself. Watch the
    // ratio of "SAVED" to "already saved" lines in the log to judge whether 40 is the right level;
    // src/test/.../BlackwoodGpuFreshFractionHarness.java (main Eternity repo) measures it properly.
    private static final int FRESH_FRACTION_PERCENT =
            parseIntEnv("ETERNITY_GPU_FRESH_FRACTION", 40);
    // Candidates to score before ranking. Scoring runs HoleSolver once per board (~1s each), and
    // only at an epoch boundary, so this bounds a startup cost rather than a per-launch one.
    private static final int MAX_SEED_CANDIDATES = 120;
    private static final boolean SEEDING_ENABLED =
            !"false".equalsIgnoreCase(System.getenv("ETERNITY_GPU_SEEDING"));
    // Draw the seed pool by depth-weighted random sampling rather than a strict top-K by depth.
    // Strict top-K made the pool a pure function of what was on disk, so every restart resumed from
    // the same elite boards: of 18 saved 12-conflict boards, only 9 were distinct, and every
    // duplicate spanned a restart. `false` restores the old deterministic selection for A/B.
    private static final boolean SEED_SAMPLING_ENABLED =
            !"false".equalsIgnoreCase(System.getenv("ETERNITY_GPU_SEED_SAMPLING"));
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

    private static boolean isEnvSet(String name) {
        String v = System.getenv(name);
        return v != null && !v.isBlank();
    }

    // Trials for HoleSolver's completion pass when scoring a candidate save (see trySave).
    // Matches the C# solver's own established choice (Util.cs's TryLabelWithConflictCount) rather
    // than HoleSolver's much heavier 200,000-trial CLI default -- this runs in the same thread as
    // the main launch loop, only on the (already rare) event of a new per-run depth record, but
    // still shouldn't stall launches for longer than that cadence tolerates.
    private static final int SCORING_TRIALS = 5000;
    private static final int HARVEST_INTERVAL = 300;
    private static final int HARVEST_SAMPLE = 100;
    private static final int HARVEST_MIN_DEPTH = 240;
    /** Boards already scored, so a stable population isn't re-scored every harvest. */
    private static final Set<String> harvestedFingerprints = new HashSet<>();
    private static final int HARVEST_MEMORY_CAP = 200_000;
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
     *
     * <p>Persisted to {@link #SAVED_BOARDS_INDEX} in the output directory, because being
     * in-memory-only made it useless across restarts -- the exact case that matters here, since the
     * process restarts on every driver TDR. Of 18 saved 12-conflict boards, only 9 were distinct
     * and every duplicate spanned a restart. Kept as hex SHA-256 of the bucas link rather than the
     * link itself: 64 bytes per board instead of ~1.5 KB, and the file never needs to be read by a
     * human.</p>
     */
    private static final Set<String> savedCompletedBoards = new HashSet<>();
    /** One hex SHA-256 per line, appended as boards are saved. */
    private static final String SAVED_BOARDS_INDEX = ".saved_completed_boards";

    public static void main(String[] args) throws Exception {
        String configuredDir = System.getenv("ETERNITY_GPU_SOLUTIONS_DIR");
        Path outputDir = (configuredDir == null || configuredDir.isBlank())
                ? Path.of(System.getProperty("user.home"), "EternitySolutions_GPU")
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
        loadSavedBoardIndex(outputDir);

        logger.info("BlackwoodGpuRunner starting. numThreads={}, initialStepBudget={}, saveThreshold={}, epochLaunches={}, resumedHighScore={}, seedingEnabled={}, sharedCacheEnabled={}, breakIndexesAllowed={}",
                NUM_THREADS, INITIAL_STEP_BUDGET, SAVE_THRESHOLD, EPOCH_LAUNCHES, currentHighScore, SEEDING_ENABLED, SHARED_CACHE_ENABLED,
                java.util.Arrays.toString(BwUtil.BREAK_INDEXES_ALLOWED));
        if (!NUM_THREADS_EXPLICIT) {
            logger.warn("ETERNITY_GPU_NUM_THREADS is not set -- running with the conservative code "
                    + "default of {} threads, not the production-tuned 16384 (see README). Launch via "
                    + "run-gpu.cmd, or set the env var explicitly, unless this is intentional.",
                    NUM_THREADS_DEFAULT);
        }

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
    private static String sha256Hex(String s) {
        try {
            byte[] digest = java.security.MessageDigest.getInstance("SHA-256")
                    .digest(s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                sb.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
            }
            return sb.toString();
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e); // every JRE ships it
        }
    }

    /**
     * Repopulates {@link #savedCompletedBoards} from disk so boards found in an earlier run are not
     * saved and uploaded again. A missing or unreadable index is not fatal -- it just means this run
     * starts with no memory, which is exactly the old behaviour.
     */
    private static void loadSavedBoardIndex(Path outputDir) {
        Path index = outputDir.resolve(SAVED_BOARDS_INDEX);
        if (!Files.isRegularFile(index)) {
            logger.info("No saved-board index at {} yet; duplicate suppression starts empty", index);
            return;
        }
        try {
            List<String> lines = Files.readAllLines(index);
            for (String line : lines) {
                String trimmed = line.trim();
                if (!trimmed.isEmpty()) savedCompletedBoards.add(trimmed);
            }
            logger.info("Loaded {} previously saved board fingerprint(s) from {}", savedCompletedBoards.size(), index);
        } catch (IOException e) {
            logger.warn("Could not read saved-board index {}; duplicate suppression starts empty", index, e);
        }
    }

    /** Never fatal: failing to record a fingerprint costs a future duplicate, not the saved board. */
    private static void appendSavedBoardHash(Path outputDir, String hash) {
        try {
            Files.writeString(outputDir.resolve(SAVED_BOARDS_INDEX), hash + System.lineSeparator(),
                    java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.warn("Could not append to saved-board index; this board may be re-saved after a restart", e);
        }
    }

    private static void loadSeeds(BlackwoodGpuEngine engine, int[] stepBoardIdx, Path gpuOutputDir,
                                  PieceInventory inventory) {
        try {
            Path home = Path.of(System.getProperty("user.home"));
            List<Path> dirs = List.of(
                    gpuOutputDir,
                    home.resolve("EternitySolutions_CSharpCPU"),   // C# solver (was split across
                                                                    // EternitySolutions + _drop239;
                                                                    // consolidated 2026-08-30)
                    home.resolve("EternitySolutions_JavaCPU"));    // Java CPU port (was wrongly pointed
                                                                    // at ~/Documents -- that path never
                                                                    // existed; fixed 2026-08-30)

            List<BwSeedLoader.Seed> candidates = BwSeedLoader.load(dirs, MIN_SEED_DEPTH,
                    MAX_SEED_CANDIDATES, stepBoardIdx,
                    SEED_SAMPLING_ENABLED ? new java.util.Random() : null);
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
            int shallowestSeed = seeds.stream().mapToInt(BwSeedLoader.Seed::depth).min().orElse(-1);
            int deepestSeed = seeds.stream().mapToInt(BwSeedLoader.Seed::depth).max().orElse(-1);
            logger.info("Seeding from {} of {} candidate board(s): conflicts {}..{}, depth {}..{}, best is {} pieces -> {} conflicts ({}). sampling={}, maxRetreat={}, freshFraction={}%",
                    seeds.size(), candidates.size(), best.conflicts(), worst.conflicts(),
                    shallowestSeed, deepestSeed,
                    best.depth(), best.conflicts(), best.source().getFileName(),
                    SEED_SAMPLING_ENABLED, MAX_RETREAT, FRESH_FRACTION_PERCENT);
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
            int keepThreshold = (bestOnDisk == Integer.MAX_VALUE)
                    ? Integer.MAX_VALUE : Math.max(ALWAYS_SAVE_AT_OR_BELOW, bestOnDisk + 1);
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
            String boardHash = sha256Hex(completedLink);
            if (!savedCompletedBoards.add(boardHash)) {
                logger.info("Completed board at {} conflicts already saved (possibly in an earlier run), skipping duplicate",
                        conflicts);
                return conflicts;
            }
            appendSavedBoardHash(outputDir, boardHash);

            Files.createDirectories(outputDir);
            String timeId = LocalTime.now().format(DateTimeFormatter.ofPattern("HHmmss_SSS"));
            String prefix = "Errors" + conflicts + "_Base" + maxSolveIndex + "_" + timeId;
            HoleSolver.writePhysicalLayoutFile(outputDir.resolve(prefix + "_physical_layout.txt").toString(),
                    inventory, result.finalBoard(), result.repairedBoard());
            HoleSolver.writeRawBoardFile(outputDir.resolve(prefix + "_RawBoard.txt").toString(), inventory, completed);
            Files.writeString(outputDir.resolve(prefix + "_baseboard.txt"), boardString);
            logger.info("SAVED [{}]: {} pieces, {} conflicts -> {}, exact={}, budgetExhausted={}",
                    depthRecord ? "depth-record" : "harvest", maxSolveIndex, conflicts, prefix, exact, budgetExhausted);
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

            int minOnDisk = conflictsByFile.values().stream().mapToInt(Integer::intValue).min().orElse(0);
            int keepThreshold = Math.max(ALWAYS_SAVE_AT_OR_BELOW, minOnDisk + 1);
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
