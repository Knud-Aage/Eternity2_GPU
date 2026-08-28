package dk.puzzle.blackwood;

import dk.puzzle.core.PieceLoader;
import dk.puzzle.io.drive.DriveUploader;
import dk.puzzle.io.BucasExporter;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Faithful port of Blackwood's {@code Program.cs} — table orchestration, the
 * per-attempt chronological-backtracking search, and the outer batch loop.
 * Runs entirely in his own raw colour numbering; see BwUtil/BwPiece.
 */
public class BlackwoodSolver {

    private static final Logger logger = LogManager.getLogger(BlackwoodSolver.class);

    // 2026-08-20 (ab_break9_cap25/cap100 vs ab_break9, all 8h): 25B/50B/100B tied on best (14) and
    // mean/median (~16.9/17) -- no quality difference found, but that 4x range never got anywhere
    // near GPU's persistent-lineage scale (~164 trillion nodes/epoch, 3000x+ beyond 50B), so it
    // doesn't settle whether a much larger cap matters. Overridable so that can actually be tested.
    static final long DEFAULT_NODE_CAP = parseLongEnv("ETERNITY_NODE_CAP", 50_000_000_000L); // matches C# `node_count > 50000000000`
    // 2026-08-19: raised 190 -> 248 to match the C# solver's own Program.cs, which made the same
    // change after 190 produced 389,856 boards (377K of them below 248, averaging 18-19 conflicts
    // vs the 12-conflict record) that flooded OneDrive and that nothing downstream reads -- the
    // conflict tracker's own floor is 248. Keep these two numbers in step (their comment says the
    // same thing back).
    private static final int DEFAULT_SAVE_THRESHOLD = 248;
    private static final int ATTEMPTS_PER_WORKER_PER_BATCH = 5;
    // Matches the C# solver's / GPU runner's own trial count for HoleSolver's completion pass.
    private static final int SCORING_TRIALS = 5000;
    // 2026-08-19: labelled save format, matching the GPU runner and C# solver -- conflicts first
    // in the name so the three engines' output is directly comparable at a glance, and so
    // BwSeedLoader (which already recognizes this exact pattern) can use this port's own best
    // boards as seeds elsewhere.
    private static final Pattern LABELLED_NAME = Pattern.compile("^Errors(\\d+)_Base(\\d+)_.*_RawBoard\\.txt$");
    // 2026-08-19: dedicated links-only log, mirroring BlackwoodGpuRunner's COMPLETED_LINKS_LOG --
    // same reasoning, same format, so both are grep-able the same way.
    private static final Path COMPLETED_LINKS_LOG = Path.of("logs", "java_port_completed_links.log");

    private final int saveThreshold;
    private final Path outputDir;
    private final int numWorkers;
    private final String piecesFilePath;

    private List<BwPiece> boardPieces;
    private BwPiece[] pieceByNumber; // index = pieceNumber, length 257
    private PieceInventory inventory; // HoleSolver's own representation, built once alongside boardPieces
    // Concurrent: evaluateAndMaybeSave can be called from any of numWorkers worker threads at once.
    private final Set<String> savedCompletedBoards = ConcurrentHashMap.newKeySet();

    // Rebuilt by prepare() once per outer batch; read-only for that batch's lifetime.
    // Safe publication relies on all workers being submit()'d only after prepare()
    // fully returns (ExecutorService.submit()'s happens-before guarantee) -- no
    // volatile/synchronized needed, matching how EternitySolver.CpuSearchWorker
    // already relies on the same pattern in this codebase.
    //
    // Package-private (not private): dk.puzzle.blackwood.BwGpuTables (2026-08-02)
    // reuses these directly to flatten into GPU CSR form rather than re-deriving
    // table construction a second time -- the single biggest fidelity-risk
    // reducer for the GPU port, since the kernel then sees exactly the same
    // tables this already-verified CPU port trusts.
    BwRotatedPiece[][] corners;
    BwRotatedPiece[][] leftSides;
    BwRotatedPiece[][] topSides;
    BwRotatedPiece[][] rightSidesWithBreaks;
    BwRotatedPiece[][] rightSidesWithoutBreaks;
    BwRotatedPiece[][] middlesWithBreak;
    BwRotatedPiece[][] middlesNoBreak;
    BwRotatedPiece[][] southStart;
    BwRotatedPiece[][] westStart;
    BwRotatedPiece[][] start;
    Map<Integer, List<BwUtil.RotatedCandidate>> bottomSidePiecesRotated; // raw, re-sorted every attempt
    BwRotatedPiece[][][] masterPieceLookup;
    int[] boardOrderRow;
    int[] boardOrderCol;
    int[] breakArray;
    int[] heuristicArray;

    // Verification instrumentation -- see plan's "solve_index==0 edge case" note.
    private final AtomicLong exhaustedAtSeedCount = new AtomicLong();

    public BlackwoodSolver() {
        this(DEFAULT_SAVE_THRESHOLD, defaultOutputDir(), defaultWorkerCount(), "src/main/resources/JBlackwood_Pieces.txt");
    }

    public BlackwoodSolver(int saveThreshold, Path outputDir, int numWorkers, String piecesFilePath) {
        this.saveThreshold = saveThreshold;
        this.outputDir = outputDir;
        this.numWorkers = numWorkers;
        this.piecesFilePath = piecesFilePath;
    }

    private static Path defaultOutputDir() {
        // 2026-08-19: moved off Documents\... -- Documents is OneDrive-redirected by Known Folder
        // Move on this machine (see commit 9286a98, "Move C# board output off OneDrive": 389,856
        // synced files, ~1GB, filled the quota before anyone noticed). UserProfile itself is never
        // touched by KFM, so this stays local -- same convention as the GPU runner's
        // ~/EternitySolutions_GpuBlackwood and the C# solver's ~/EternitySolutions, just with its
        // own suffix so provenance of any given save file is still unambiguous.
        return Path.of(System.getProperty("user.home"), "EternitySolutions_JavaPort");
    }

    private static int defaultWorkerCount() {
        return Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
    }

    long exhaustedAtSeedCount() {
        return exhaustedAtSeedCount.get();
    }

    /** Mirrors Prepare_Pieces_And_Heuristics(). Single-threaded; call before launching a batch of workers. */
    void prepare() throws Exception {
        if (boardPieces == null) {
            boardPieces = BwUtil.getPieces(piecesFilePath);
            pieceByNumber = new BwPiece[257];
            for (BwPiece p : boardPieces) {
                pieceByNumber[p.pieceNumber()] = p;
            }
            // HoleSolver's own representation, needed by evaluateAndMaybeSave's completion pass --
            // separate from Blackwood's raw numbering above, built once since it never changes.
            inventory = new PieceInventory(PieceLoader.loadPieces());
        }

        List<BwPiece> cornerPieces = boardPieces.stream().filter(p -> p.pieceType() == 2).toList();
        List<BwPiece> sidePieces = boardPieces.stream().filter(p -> p.pieceType() == 1).toList();
        List<BwPiece> middlePieces = boardPieces.stream().filter(p -> p.pieceType() == 0 && p.pieceNumber() != 139).toList();
        BwPiece startPiece = boardPieces.stream().filter(p -> p.pieceNumber() == 139).findFirst().orElseThrow();

        Random rand = new Random(); // one instance reused across this whole prepare() call, matches C#'s lifetime

        corners = buildTable(cornerPieces, false, null, rand);

        List<BwUtil.RotatedCandidate> sidesNoBreak = new ArrayList<>();
        for (BwPiece p : sidePieces) {
            sidesNoBreak.addAll(BwUtil.getRotatedPieces(p, false));
        }
        List<BwUtil.RotatedCandidate> sidesWithBreak = new ArrayList<>();
        for (BwPiece p : sidePieces) {
            sidesWithBreak.addAll(BwUtil.getRotatedPieces(p, true));
        }

        bottomSidePiecesRotated = BwUtil.groupByLeftBottom(filterRotation(sidesNoBreak, 0)); // raw, unsorted
        leftSides = BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(filterRotation(sidesNoBreak, 1)), rand);
        rightSidesWithoutBreaks = BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(filterRotation(sidesNoBreak, 3)), rand);
        topSides = BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(filterRotation(sidesWithBreak, 2)), rand);
        rightSidesWithBreaks = BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(filterRotation(sidesWithBreak, 3)), rand);

        middlesWithBreak = buildTable(middlePieces, true, null, rand);
        middlesNoBreak = buildTable(middlePieces, false, null, rand);
        southStart = buildTable(middlePieces, false, rp -> rp.topSide() == 6, rand);
        westStart = buildTable(middlePieces, false, rp -> rp.rightSide() == 11, rand);
        start = buildTable(List.of(startPiece), false, rp -> rp.rotations() == 2, rand);

        if (corners[0] == null || corners[0].length == 0) {
            throw new IllegalStateException("corners[0] is empty -- no corner piece qualifies for LeftBottom=0; step-0 seeding would fail.");
        }

        BwUtil.BoardOrder order = BwUtil.getBoardOrder();
        boardOrderRow = order.rows();
        boardOrderCol = order.cols();
        breakArray = BwUtil.getBreakArray();
        heuristicArray = BwUtil.getHeuristicArray();

        int firstBreakIndex = BwUtil.firstBreakIndex();
        masterPieceLookup = new BwRotatedPiece[256][][];
        for (int i = 0; i < 256; i++) {
            int row = boardOrderRow[i];
            int col = boardOrderCol[i];
            if (row == 15) {
                masterPieceLookup[row * 16 + col] = (col == 15 || col == 0) ? corners : topSides;
            } else if (row == 0) {
                // Deliberately left null -- row 0 handled specially in solvePuzzle() via bottomSides/corners.
            } else if (col == 15) {
                masterPieceLookup[row * 16 + col] = (i < firstBreakIndex) ? rightSidesWithoutBreaks : rightSidesWithBreaks;
            } else if (col == 0) {
                masterPieceLookup[row * 16 + col] = leftSides;
            } else if (row == 7 && col == 7) {
                masterPieceLookup[row * 16 + col] = start;
            } else if (row == 7 && col == 6) {
                masterPieceLookup[row * 16 + col] = westStart;
            } else if (row == 6 && col == 7) {
                masterPieceLookup[row * 16 + col] = southStart;
            } else {
                masterPieceLookup[row * 16 + col] = (i < firstBreakIndex) ? middlesNoBreak : middlesWithBreak;
            }
        }
    }

    private BwRotatedPiece[][] buildTable(List<BwPiece> pieces, boolean allowBreaks, Predicate<BwRotatedPiece> filter, Random rand) {
        List<BwUtil.RotatedCandidate> all = new ArrayList<>();
        for (BwPiece piece : pieces) {
            all.addAll(BwUtil.getRotatedPieces(piece, allowBreaks));
        }
        if (filter != null) {
            all = all.stream().filter(c -> filter.test(c.rotatedPiece())).toList();
        }
        return BwUtil.sortAndFreezeByScore(BwUtil.groupByLeftBottom(all), rand);
    }

    private static List<BwUtil.RotatedCandidate> filterRotation(List<BwUtil.RotatedCandidate> list, int rotation) {
        return list.stream().filter(c -> c.rotatedPiece().rotations() == rotation).toList();
    }

    /**
     * @param lastImprovementNode nodeCount at which maxSolveIndex last increased. Pure
     *   instrumentation for the 2026-08-24 stagnation study: the gap between this and nodeCount is
     *   how long the attempt ran without getting any deeper, which is what a stagnation-based
     *   restart rule would need to be tuned against. Measured before building the rule rather
     *   than guessing a threshold.
     */
    /**
     * @param maxLateGap largest run of nodes that passed with NO depth gain and was then followed by
     *   a gain reaching depth >= {@link #LATE_DEPTH}. This is the number a stagnation-based restart
     *   threshold must exceed: set the threshold below this and the rule would have killed an
     *   attempt during a quiet spell that was, in fact, about to produce a deep improvement.
     *   {@code lastImprovementNode} alone can't show this -- it only measures the tail after the
     *   final gain, not the dry spells between gains.
     */
    public record SolveResult(int maxSolveIndex, BwRotatedPiece[] board, long nodeCount, boolean completed,
                              long lastImprovementNode, long maxLateGap) {
    }

    /** Depth past which an improvement counts as "late" for {@link SolveResult#maxLateGap}. */
    static final int LATE_DEPTH = 240;

    /** Java-only guard: the general row==0,col==0 path would otherwise read board[-1] (see plan's edge-case note). */
    static boolean attemptExhausted(int solveIndex) {
        return solveIndex < 1;
    }

    SolveResult solvePuzzle() {
        return solvePuzzle(DEFAULT_NODE_CAP);
    }

    /** Mirrors SolvePuzzle(). nodeCap is injectable for bounded tests; production uses DEFAULT_NODE_CAP. */
    SolveResult solvePuzzle(long nodeCap) {
        boolean[] pieceUsed = new boolean[257];
        int[] cumulativeHeuristicSideCount = new int[256];
        int[] pieceIndexToTryNext = new int[256];
        int[] cumulativeBreaks = new int[256];
        BwRotatedPiece[] board = new BwRotatedPiece[256];
        Arrays.fill(board, BwRotatedPiece.EMPTY);

        Random rand = new Random(); // fresh per attempt

        BwRotatedPiece[][] bottomSides = BwUtil.sortAndFreezeBottomSides(bottomSidePiecesRotated, rand);

        BwRotatedPiece[] cornerZero = corners[0];
        board[0] = cornerZero[rand.nextInt(cornerZero.length)]; // uniform pick -- see plan text for why this is faithful
        pieceUsed[board[0].pieceNumber()] = true;
        cumulativeBreaks[0] = 0;
        cumulativeHeuristicSideCount[0] = board[0].heuristicSideCount();

        int solveIndex = 1;
        int maxSolveIndex = solveIndex;
        long nodeCount = 0;
        long lastImprovementNode = 0;
        long maxLateGap = 0;

        while (true) {
            nodeCount++;

            if (solveIndex > maxSolveIndex) {
                // Measure the quiet spell BEFORE overwriting lastImprovementNode. Only gains that
                // land deep count: the early climb to ~240 is near-instant and its sub-second gaps
                // would drown out the late-stage dry spells that actually set the threshold.
                if (solveIndex >= LATE_DEPTH) {
                    long gap = nodeCount - lastImprovementNode;
                    if (gap > maxLateGap) maxLateGap = gap;
                }
                maxSolveIndex = solveIndex;
                lastImprovementNode = nodeCount;
                if (maxSolveIndex >= saveThreshold) {
                    evaluateAndMaybeSave(board, maxSolveIndex);
                    if (maxSolveIndex >= 256) {
                        return new SolveResult(maxSolveIndex, board, nodeCount, true, lastImprovementNode, maxLateGap);
                    }
                }
            }

            if (nodeCount > nodeCap) {
                return new SolveResult(maxSolveIndex, board, nodeCount, false, lastImprovementNode, maxLateGap);
            }

            if (attemptExhausted(solveIndex)) {
                exhaustedAtSeedCount.incrementAndGet();
                return new SolveResult(maxSolveIndex, board, nodeCount, false, lastImprovementNode, maxLateGap);
            }

            int row = boardOrderRow[solveIndex];
            int col = boardOrderCol[solveIndex];

            if (board[row * 16 + col].pieceNumber() > 0) {
                pieceUsed[board[row * 16 + col].pieceNumber()] = false;
                board[row * 16 + col] = BwRotatedPiece.EMPTY;
            }

            BwRotatedPiece[] candidates;
            if (row == 0) {
                candidates = (col < 15)
                        ? bottomSides[board[row * 16 + (col - 1)].rightSide() * 23]
                        : corners[board[row * 16 + (col - 1)].rightSide() * 23];
            } else {
                int leftSide = (col == 0) ? 0 : board[row * 16 + (col - 1)].rightSide();
                candidates = masterPieceLookup[row * 16 + col][leftSide * 23 + board[(row - 1) * 16 + col].topSide()];
            }

            boolean foundPiece = false;
            if (candidates != null) {
                int breaksThisTurn = breakArray[solveIndex] - cumulativeBreaks[solveIndex - 1];
                int tryIndex = pieceIndexToTryNext[solveIndex];

                for (int i = tryIndex; i < candidates.length; i++) {
                    if (candidates[i].breakCount() > breaksThisTurn) {
                        break;
                    }

                    if (!pieceUsed[candidates[i].pieceNumber()]) {
                        if (solveIndex <= BwUtil.MAX_HEURISTIC_INDEX) {
                            if ((cumulativeHeuristicSideCount[solveIndex - 1] + candidates[i].heuristicSideCount())
                                    < heuristicArray[solveIndex]) {
                                break; // abandons the WHOLE scan for this solveIndex, not just this candidate
                            }
                        }

                        foundPiece = true;
                        BwRotatedPiece piece = candidates[i];
                        board[row * 16 + col] = piece;
                        pieceUsed[piece.pieceNumber()] = true;
                        cumulativeBreaks[solveIndex] = cumulativeBreaks[solveIndex - 1] + piece.breakCount();
                        cumulativeHeuristicSideCount[solveIndex] = cumulativeHeuristicSideCount[solveIndex - 1] + piece.heuristicSideCount();
                        pieceIndexToTryNext[solveIndex] = i + 1;
                        solveIndex++;
                        break;
                    }
                }
            }

            if (!foundPiece) {
                pieceIndexToTryNext[solveIndex] = 0;
                solveIndex--;
            }
        }
    }

    /**
     * Completes the board via HoleSolver, and saves it labelled with its real conflict count --
     * matching the GPU runner's own evaluateAndMaybeSave and the C# solver's TryLabelWithConflictCount,
     * so all three engines' output is directly comparable and equally disk-disciplined. Only keeps
     * boards within 1 conflict of whatever is already the best on disk (see pruneAboveThreshold) --
     * without this gate, this path reproduces exactly the flood that moved the output off OneDrive
     * in the first place, just with fancier filenames.
     */
    private void evaluateAndMaybeSave(BwRotatedPiece[] board, int maxSolveIndex) {
        try {
            String boardString = BwUtil.buildBoardString(board, pieceByNumber);
            String link = boardString.substring(boardString.lastIndexOf("https://"));

            int[] decoded = HoleSolver.decodeBoardAuto(link, inventory, false);
            HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(decoded, inventory, false, SCORING_TRIALS);
            int[] completed = result.bestBoard();
            int conflicts = countConflicts(completed);

            // 2026-08-28: see BlackwoodGpuRunner's identical comment -- measures how often
            // HoleSolver's exact MRV completion clears every region vs. falling back to MCV
            // heuristic repair, and (via budgetExhausted) whether that's an inconclusive
            // budget timeout or a proven dead end, to decide whether an adaptive-rewind tail
            // is worth building.
            boolean exact = result.repairedBoard() == null;
            boolean budgetExhausted = result.anyRegionBudgetExhausted();
            int bestOnDisk = bestConflictsOnDisk(outputDir);
            int keepThreshold = (bestOnDisk == Integer.MAX_VALUE) ? Integer.MAX_VALUE : bestOnDisk + 1;
            if (conflicts > keepThreshold) {
                logger.info("Depth record at {} pieces completed to {} conflicts -- not within 1 of best-on-disk ({}), not saving, exact={}, budgetExhausted={}",
                        maxSolveIndex, conflicts, bestOnDisk, exact, budgetExhausted);
                return;
            }

            String completedLink = BucasExporter.exportBoard(completed);
            if (!savedCompletedBoards.add(completedLink)) {
                logger.debug("Completed board at {} conflicts already saved, skipping duplicate", conflicts);
                return;
            }

            Files.createDirectories(outputDir);
            String timeId = LocalTime.now().format(DateTimeFormatter.ofPattern("HHmmss_SSS"));
            String prefix = "Errors" + conflicts + "_Base" + maxSolveIndex + "_" + timeId;
            HoleSolver.writePhysicalLayoutFile(outputDir.resolve(prefix + "_physical_layout.txt").toString(),
                    inventory, result.finalBoard(), result.repairedBoard());
            HoleSolver.writeRawBoardFile(outputDir.resolve(prefix + "_RawBoard.txt").toString(), inventory, completed);
            // Third sibling, in Blackwood's own piece numbering -- what BwSeedLoader can actually
            // read as a future seed, same rationale as the GPU runner's own baseboard file.
            Files.writeString(outputDir.resolve(prefix + "_baseboard.txt"), boardString);
            logger.info("Saved new personal best [depth-record]: {} pieces, {} conflicts -> {}, exact={}, budgetExhausted={}", maxSolveIndex, conflicts, prefix, exact, budgetExhausted);
            appendCompletedLink(prefix, conflicts, maxSolveIndex, completedLink);
            DriveUploader.uploadRecord(prefix, conflicts, completedLink, "Java-CPU");

            pruneAboveThreshold(outputDir, conflicts);
        } catch (Exception e) {
            logger.error("Failed to evaluate/save board at solveIndex={}", maxSolveIndex, e);
        }
    }

    /**
     * Appends one line to {@link #COMPLETED_LINKS_LOG}. Best-effort: a failure here must never
     * affect the save that already succeeded. Direct file write, not a log4j appender -- same
     * reasoning as the GPU runner's own appendCompletedLink.
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

    private static long parseLongEnv(String name, long defaultValue) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) return defaultValue;
        try {
            return Long.parseLong(v.trim());
        } catch (NumberFormatException e) {
            System.err.println("Ignoring invalid " + name + "=" + v + ", using default " + defaultValue);
            return defaultValue;
        }
    }

    private static int countConflicts(int[] board) {
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

    /** Keeps only boards within 1 conflict of the best currently on disk -- mirrors the GPU runner's own policy. */
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
                Path baseboardFile = rawBoardFile.resolveSibling(prefix + "_baseboard.txt");
                Files.deleteIfExists(rawBoardFile);
                Files.deleteIfExists(layoutFile);
                Files.deleteIfExists(baseboardFile);
            }
        } catch (IOException e) {
            logger.warn("Retention cleanup failed in {}", outputDir, e);
        }
    }

    /** Mirrors Main()'s outer while(true) + Parallel.For. */
    public void run() {
        while (true) {
            try {
                prepare();
            } catch (Exception e) {
                logger.error("prepare() failed", e);
                return;
            }
            logger.info("Tables rebuilt; {} attempts queued across {} workers. breakIndexesAllowed={}",
                    numWorkers * ATTEMPTS_PER_WORKER_PER_BATCH, numWorkers,
                    Arrays.toString(BwUtil.BREAK_INDEXES_ALLOWED));

            // 2026-08-24: shared work queue rather than a fixed slice of attempts per worker.
            //
            // The old shape was `for each of numWorkers: submit(() -> do exactly N attempts)`, which
            // puts a barrier at the end of every batch: a worker that finishes its N early has
            // nothing left to do and idles until the SLOWEST worker finishes its Nth. Attempt
            // durations vary a lot (they all retire ~50B nodes, but not at the same rate), so that
            // tail is not small. Measured over 8 complete batches in logs/java_port.log: mean 11.5%
            // of all worker-time idle, and growing -- pool 7 wasted 4.2% with a 13-minute tail,
            // pool 14 wasted 16.0% with a 2.6-HOUR tail.
            //
            // Same total attempts per batch and the same table-rebuild cadence; the only change is
            // that a free worker pulls the next attempt instead of waiting. The batch still ends
            // when all attempts are done, so tables are never rebuilt under a running search
            // (which would invalidate the candidate tables a worker is mid-way through reading).
            int attemptsThisBatch = numWorkers * ATTEMPTS_PER_WORKER_PER_BATCH;
            ExecutorService executor = Executors.newFixedThreadPool(numWorkers);
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < attemptsThisBatch; i++) {
                futures.add(executor.submit(() -> {
                    SolveResult r = solvePuzzle();
                    // stagnationNodes = how far this attempt ran after its LAST depth gain.
                    // Kept from the 2026-08-24 stagnation study: measured, found not actionable
                    // (deep gains follow LONG quiet spells), retained because it costs nothing --
                    // period-over-period throughput was unchanged with it enabled.
                    logger.info("Attempt done: maxSolveIndex={} nodeCount={} completed={} lastImprovementNode={} stagnationNodes={} maxLateGap={}",
                            r.maxSolveIndex(), r.nodeCount(), r.completed(),
                            r.lastImprovementNode(), r.nodeCount() - r.lastImprovementNode(), r.maxLateGap());
                }));
            }
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    logger.error("Attempt failed", e);
                }
            }
            executor.shutdown();
            logger.info("Batch complete ({} attempts). exhaustedAtSeedCount so far = {}",
                    attemptsThisBatch, exhaustedAtSeedCount());
        }
    }

    public static void main(String[] args) {
        new BlackwoodSolver().run();
    }
}
