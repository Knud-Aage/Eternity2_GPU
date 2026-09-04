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

public class BlackwoodSolver {

    private static final Logger logger = LogManager.getLogger(BlackwoodSolver.class);

    static final long DEFAULT_NODE_CAP = parseLongEnv("ETERNITY_NODE_CAP", 50_000_000_000L); // matches C# `node_count > 50000000000`
    private static final int DEFAULT_SAVE_THRESHOLD = 248;
    // See BlackwoodGpuRunner's ALWAYS_SAVE_AT_OR_BELOW for the full rationale: the save/retention
    // window is normally "within 1 of best-on-disk", which tightens forever as the record improves.
    // This floor keeps <=12 permanently save-worthy regardless. Mirror any change in
    // BlackwoodGpuRunner.java, the CPU repo's copy of this file, and Util.cs's PruneAboveThreshold.
    private static final int ALWAYS_SAVE_AT_OR_BELOW = parseIntEnv("ETERNITY_SAVE_FLOOR", 12);
    private static final int ATTEMPTS_PER_WORKER_PER_BATCH = 5;
    private static final int SCORING_TRIALS = 5000;

    // 2026-09-02: runtime switch for the 4 non-center official clues, so both modes can run from
    // one branch instead of maintaining separate "hints"/"no hints" branches. Defaults OFF to
    // match main/master's pre-existing behaviour -- set ETERNITY_NON_CENTER_HINTS=true to enable.
    // The center pin (139) predates this switch and this whole 4-hint feature; it stays
    // unconditional either way, matching what main/master already did before any of this.
    // Deliberately NOT final -- BlackwoodSolverHintPinsTest overrides it for the duration of that
    // test class (System.getenv() can't be overridden from within the same JVM), then restores
    // the default in @AfterAll so it can't leak into other test classes.
    static boolean NON_CENTER_HINTS_ENABLED =
            "true".equalsIgnoreCase(System.getenv("ETERNITY_NON_CENTER_HINTS"));

    /**
     * The official Eternity II clue: piece number, board position (0-indexed), and required
     * rotation. Position and rotation were NOT taken from any public writeup -- they were
     * independently re-derived 2026-09-01.
     * <p><b>2026-09-01 correction:</b> the initial derivation swapped 208 with 181 and 255 with
     * 249 -- each pair's two POSITIONS were correct as a set, but which piece went with which of
     * the two was backwards. Confirmed via bucas's own "Clues" preset, which encodes an explicit
     * {@code board_pieces} piece-number-per-cell list (no colour decoding involved, unlike
     * {@code board_edges}) -- searching it directly for these 5 piece numbers gives their true
     * positions unambiguously. The bug was caught because it made both wrongly-placed pieces
     * nearly unplaceable: GPU search plateaued hard at depth 200 and the CPU/C# ports couldn't get
     * past depth 188 (181's old, wrong cell) in any attempt. Rotations are unaffected by this fix
     * -- they were derived per PIECE, not per position (see the retired EternitySolver.java's
     * {@code exactHintIds}/{@code exactHintRots} arrays, which pair each rotation with a piece
     * number), so relocating a piece to its correct cell keeps its already-derived rotation.
     * Rotation itself was cross-checked two ways when first derived: (1) it was translated by
     * finding a CONSTANT +2 offset between TheSil's colour space and Blackwood's raw one (via
     * {@code BLACKWOOD_TO_THESIL}), holding with zero exceptions across all 4 non-center pieces;
     * (2) as a blind check, the same +2 offset predicts rotation 2 for the center (139) too,
     * exactly matching the `rotations() == 2` filter already live below, which that derivation did
     * not use as an input. A THIRD, independent re-verification of rotation (decoding the "Clues"
     * preset's own board_edges colours) was attempted alongside the position fix above but hit an
     * unresolved colour-scheme mismatch even on the known-good center piece, so it could not
     * confirm or refute rotation -- position is solid, rotation rests on the original two-way
     * check only.
     */
    private record HintPin(int pieceNumber, int row, int col, int rotation) {
    }

    private static final List<HintPin> HINT_PINS = List.of(
            new HintPin(139, 7, 7, 2),   // center
            new HintPin(181, 2, 2, 2),
            new HintPin(249, 2, 13, 3),
            new HintPin(208, 13, 2, 2),
            new HintPin(255, 13, 13, 2));
    private static final Pattern LABELLED_NAME = Pattern.compile("^Errors(\\d+)_Base(\\d+)_.*_RawBoard\\.txt$");
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

    BwRotatedPiece[][] corners;
    BwRotatedPiece[][] leftSides;
    BwRotatedPiece[][] topSides;
    BwRotatedPiece[][] rightSidesWithBreaks;
    BwRotatedPiece[][] rightSidesWithoutBreaks;
    BwRotatedPiece[][] middlesWithBreak;
    BwRotatedPiece[][] middlesNoBreak;
    BwRotatedPiece[][] southStart;
    BwRotatedPiece[][] westStart;
    BwRotatedPiece[][] start; // piece 139, the center clue
    // The 4 non-center clues. Named individually, not held in a Map, because BwGpuTables.build()
    // matches masterPieceLookup entries against tablesInOrder() by REFERENCE IDENTITY -- each one
    // needs its own distinct, directly-nameable field for that to work.
    BwRotatedPiece[][] hint208;
    BwRotatedPiece[][] hint255;
    BwRotatedPiece[][] hint181;
    BwRotatedPiece[][] hint249;
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
        return Path.of(System.getProperty("user.home"), "EternitySolutions_JavaCPU");
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
        // Center (139) is excluded from the general pool unconditionally, matching main/master's
        // pre-existing behaviour. The other 4 are only pulled out when the switch is on --
        // otherwise they must stay available to the general search like any other middle piece.
        Set<Integer> hintPieceNumbers = NON_CENTER_HINTS_ENABLED
                ? HINT_PINS.stream().map(HintPin::pieceNumber).collect(java.util.stream.Collectors.toSet())
                : Set.of(139);
        List<BwPiece> middlePieces = boardPieces.stream().filter(p -> p.pieceType() == 0 && !hintPieceNumbers.contains(p.pieceNumber())).toList();
        java.util.function.Function<Integer, BwPiece> hintPiece = num ->
                boardPieces.stream().filter(p -> p.pieceNumber() == num).findFirst()
                        .orElseThrow(() -> new IllegalStateException("Hint piece " + num + " not found in piece set"));

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
        start = buildTable(List.of(hintPiece.apply(139)), false, rp -> rp.rotations() == 2, rand);
        // All four non-center hints get allowBreaks=true (see BwUtil.HINT_BREAK_INDEXES) -- 208
        // and 255 first (2026-09-02), 181 and 249 added 2026-09-04 after 181 turned out to be the
        // ACTUAL fill-step-34 hint (208 is really at step 188, 255 at 247 -- see BwUtil's
        // corrected write-up), and a hard zero-tolerance pin sitting first in fill order is a much
        // better explanation for the population bottlenecking at 34 than anything about 208. Only
        // start (139, the mandatory center) stays a hard pin -- it predates the whole hint feature
        // and has never shown this problem.
        hint208 = buildTable(List.of(hintPiece.apply(208)), true, rp -> rp.rotations() == 2, rand);
        hint255 = buildTable(List.of(hintPiece.apply(255)), true, rp -> rp.rotations() == 2, rand);
        hint181 = buildTable(List.of(hintPiece.apply(181)), true, rp -> rp.rotations() == 2, rand);
        hint249 = buildTable(List.of(hintPiece.apply(249)), true, rp -> rp.rotations() == 3, rand);

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
            } else if (NON_CENTER_HINTS_ENABLED && row == 2 && col == 2) {
                masterPieceLookup[row * 16 + col] = hint181;
            } else if (NON_CENTER_HINTS_ENABLED && row == 2 && col == 13) {
                masterPieceLookup[row * 16 + col] = hint249;
            } else if (NON_CENTER_HINTS_ENABLED && row == 13 && col == 2) {
                masterPieceLookup[row * 16 + col] = hint208;
            } else if (NON_CENTER_HINTS_ENABLED && row == 13 && col == 13) {
                masterPieceLookup[row * 16 + col] = hint255;
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

    private void evaluateAndMaybeSave(BwRotatedPiece[] board, int maxSolveIndex) {
        try {
            String boardString = BwUtil.buildBoardString(board, pieceByNumber);
            String link = boardString.substring(boardString.lastIndexOf("https://"));

            int[] decoded = HoleSolver.decodeBoardAuto(link, inventory, false);
            HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(decoded, inventory, false, SCORING_TRIALS);
            int[] completed = result.bestBoard();
            int conflicts = countConflicts(completed);

            boolean exact = result.repairedBoard() == null;
            boolean budgetExhausted = result.anyRegionBudgetExhausted();
            int bestOnDisk = bestConflictsOnDisk(outputDir);
            int keepThreshold = (bestOnDisk == Integer.MAX_VALUE)
                    ? Integer.MAX_VALUE : Math.max(ALWAYS_SAVE_AT_OR_BELOW, bestOnDisk + 1);
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

            int minOnDisk = conflictsByFile.values().stream().mapToInt(Integer::intValue).min().orElse(0);
            int keepThreshold = Math.max(ALWAYS_SAVE_AT_OR_BELOW, minOnDisk + 1);
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

            int attemptsThisBatch = numWorkers * ATTEMPTS_PER_WORKER_PER_BATCH;
            ExecutorService executor = Executors.newFixedThreadPool(numWorkers);
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < attemptsThisBatch; i++) {
                futures.add(executor.submit(() -> {
                    SolveResult r = solvePuzzle();
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
