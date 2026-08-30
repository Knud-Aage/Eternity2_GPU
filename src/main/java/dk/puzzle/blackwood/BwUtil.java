package dk.puzzle.blackwood;

import dk.puzzle.io.JBlackwoodToBucas;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.ToIntFunction;

public final class BwUtil {

    public static final int[] SIDE_EDGES = {1, 5, 9, 13, 17};
    public static final int[] HEURISTIC_SIDES = {13, 16, 10};
    // Blackwood's original 10-entry schedule. The 239 entry was dropped 2026-08-2x on the theory
    // that allowing 10 breaks capped the best possible result at 470 -- that was wrong, and it was
    // restored 2026-08-30. The array is a per-depth CEILING, not a quota: nothing obliges the search
    // to spend its allowance, and candidates are scored `score - 100000 * breakCount`, so break-free
    // placements are tried first at every step and a break is taken only when the search is stuck.
    // A 10-break-allowed run that finishes on 9 breaks is 471, perfectly legally. Dropping 239 also
    // cost reachability outright: getBreakArray() counts entries <= i, so the 9- and 10-entry arrays
    // are IDENTICAL below depth 239 and the 10-entry one is strictly more permissive above it --
    // i.e. the 10-break search tree is a superset. A leave-one-out sweep had already measured that
    // 9-break configurations reach depth 248 far more rarely, which was the price paid for nothing.
    public static final int[] BREAK_INDEXES_ALLOWED = {201, 206, 211, 216, 221, 225, 229, 233, 237, 239};
    public static final int MAX_HEURISTIC_INDEX = 160;

    public static final int[] BLACKWOOD_TO_THESIL = {
            0, 1, 6, 22, 17, 3, 8, 10, 12, 4, 7, 9, 18, 5, 15, 11, 20, 2, 14, 16, 19, 13, 21
    };

    private static final boolean[] SIDE_EDGE_LOOKUP = new boolean[23];
    static {
        for (int s : SIDE_EDGES) {
            SIDE_EDGE_LOOKUP[s] = true;
        }
    }

    private BwUtil() {
    }

    public static int calculateTwoSides(int side1, int side2) {
        return side1 * 23 + side2;
    }

    /** Transient table-construction record. Mirrors RotatedPieceWithLeftBottom; Score dropped once tables freeze. */
    public record RotatedCandidate(int leftBottom, int score, BwRotatedPiece rotatedPiece) {
    }

    public static List<RotatedCandidate> getRotatedPieces(BwPiece piece) {
        return getRotatedPieces(piece, false);
    }

    public static List<RotatedCandidate> getRotatedPieces(BwPiece piece, boolean allowBreaks) {
        int score = 0;
        int heuristicSideCount = 0;
        for (int side : HEURISTIC_SIDES) {
            if (piece.leftSide() == side) {
                score += 100;
                heuristicSideCount++;
            }
            if (piece.topSide() == side) {
                score += 100;
                heuristicSideCount++;
            }
            if (piece.rightSide() == side) {
                score += 100;
                heuristicSideCount++;
            }
            if (piece.bottomSide() == side) {
                score += 100;
                heuristicSideCount++;
            }
        }

        List<RotatedCandidate> out = new ArrayList<>();
        for (int left = 0; left <= 22; left++) {
            for (int bottom = 0; bottom <= 22; bottom++) {
                int leftBottom = calculateTwoSides(left, bottom);

                // rotation 0: west-facing=LeftSide, south-facing=BottomSide; emits Top/Right unchanged.
                addCandidateIfValid(out, piece, leftBottom, score, allowBreaks,
                        piece.leftSide(), left, piece.bottomSide(), bottom,
                        0, piece.topSide(), piece.rightSide(), heuristicSideCount);

                // rotation 1: west-facing=BottomSide, south-facing=RightSide; emits Top=Left, Right=Top.
                addCandidateIfValid(out, piece, leftBottom, score, allowBreaks,
                        piece.bottomSide(), left, piece.rightSide(), bottom,
                        1, piece.leftSide(), piece.topSide(), heuristicSideCount);

                // rotation 2: west-facing=RightSide, south-facing=TopSide; emits Top=Bottom, Right=Left.
                addCandidateIfValid(out, piece, leftBottom, score, allowBreaks,
                        piece.rightSide(), left, piece.topSide(), bottom,
                        2, piece.bottomSide(), piece.leftSide(), heuristicSideCount);

                // rotation 3: west-facing=TopSide, south-facing=LeftSide; emits Top=Right, Right=Bottom.
                addCandidateIfValid(out, piece, leftBottom, score, allowBreaks,
                        piece.topSide(), left, piece.leftSide(), bottom,
                        3, piece.rightSide(), piece.bottomSide(), heuristicSideCount);
            }
        }
        return out;
    }

    private static void addCandidateIfValid(List<RotatedCandidate> out, BwPiece piece, int leftBottom, int score,
            boolean allowBreaks, int westFacing, int requiredLeft, int southFacing, int requiredBottom,
            int rotation, int emittedTop, int emittedRight, int heuristicSideCount) {
        int breakCount = 0;
        int sideBreaks = 0;
        if (westFacing != requiredLeft) {
            breakCount++;
            if (SIDE_EDGE_LOOKUP[westFacing]) {
                sideBreaks++;
            }
        }
        if (southFacing != requiredBottom) {
            breakCount++;
            if (SIDE_EDGE_LOOKUP[southFacing]) {
                sideBreaks++;
            }
        }
        if ((breakCount == 0 || (breakCount == 1 && allowBreaks)) && sideBreaks == 0) {
            out.add(new RotatedCandidate(leftBottom, score - 100000 * breakCount,
                    new BwRotatedPiece(piece.pieceNumber(), rotation, emittedTop, emittedRight, breakCount, heuristicSideCount)));
        }
    }

    public static Map<Integer, List<RotatedCandidate>> groupByLeftBottom(List<RotatedCandidate> candidates) {
        Map<Integer, List<RotatedCandidate>> grouped = new LinkedHashMap<>();
        for (RotatedCandidate c : candidates) {
            grouped.computeIfAbsent(c.leftBottom(), k -> new ArrayList<>()).add(c);
        }
        return grouped;
    }

    /** The 9 batch-level tables: descending by (Score + rand.nextInt(99)), jitter computed once per entry. */
    public static BwRotatedPiece[][] sortAndFreezeByScore(Map<Integer, List<RotatedCandidate>> grouped, Random rand) {
        return sortAndFreeze(grouped, c -> c.score() + rand.nextInt(99));
    }

    /** bottom_sides only, rebuilt fresh every solvePuzzle() call with a DIFFERENT formula. */
    public static BwRotatedPiece[][] sortAndFreezeBottomSides(Map<Integer, List<RotatedCandidate>> grouped, Random rand) {
        return sortAndFreeze(grouped, c -> (c.rotatedPiece().heuristicSideCount() > 0 ? 100 : 0) + rand.nextInt(99));
    }

    private static BwRotatedPiece[][] sortAndFreeze(Map<Integer, List<RotatedCandidate>> grouped, ToIntFunction<RotatedCandidate> keyFn) {
        BwRotatedPiece[][] result = new BwRotatedPiece[529][];
        for (Map.Entry<Integer, List<RotatedCandidate>> e : grouped.entrySet()) {
            record Keyed(int key, RotatedCandidate candidate) {
            }
            List<Keyed> keyed = new ArrayList<>(e.getValue().size());
            for (RotatedCandidate c : e.getValue()) {
                keyed.add(new Keyed(keyFn.applyAsInt(c), c)); // key computed ONCE, not resampled during sort
            }
            keyed.sort((a, b) -> Integer.compare(b.key(), a.key())); // descending, stable (matches LINQ OrderByDescending)
            BwRotatedPiece[] arr = new BwRotatedPiece[keyed.size()];
            for (int i = 0; i < arr.length; i++) {
                arr[i] = keyed.get(i).candidate().rotatedPiece();
            }
            result[e.getKey()] = arr;
        }
        return result;
    }

    public static int firstBreakIndex() {
        int min = Integer.MAX_VALUE;
        for (int v : BREAK_INDEXES_ALLOWED) {
            min = Math.min(min, v);
        }
        return min; // 201
    }

    /** Cumulative count of unlocked break-budget steps at or before each index. Mirrors Util.Get_Break_Array(). */
    public static int[] getBreakArray() {
        int[] arr = new int[256];
        int cumulative = 0;
        for (int i = 0; i < 256; i++) {
            for (int allowed : BREAK_INDEXES_ALLOWED) {
                if (allowed == i) {
                    cumulative++;
                    break;
                }
            }
            arr[i] = cumulative;
        }
        return arr;
    }

    public static int[] getHeuristicArray() {
        int[] arr = new int[256];
        for (int i = 0; i < 256; i++) {
            if (i <= 16) {
                arr[i] = 0;
            } else if (i <= 26) {
                arr[i] = (int) (((float) i - 16) * 2.8f);
            } else if (i <= 56) {
                arr[i] = (int) ((((float) i - 26) * 1.43333f) + 28);
            } else if (i <= 76) {
                arr[i] = (int) ((((float) i - 56) * 0.9f) + 71);
            } else if (i <= 102) {
                arr[i] = (int) ((((float) i - 76) * 0.6538f) + 89);
            } else if (i <= MAX_HEURISTIC_INDEX) {
                arr[i] = (int) ((((float) i - 102) / 4.4615) + 106); // 4.4615 deliberately double, not float — see javadoc
            }
            // i > 160 stays 0 — never read (solver gates on solveIndex <= MAX_HEURISTIC_INDEX)
        }
        return arr;
    }

    public record BoardOrder(int[] rows, int[] cols) {
    }

    /** Literal transcription of Util.Get_Board_Order()'s board_order array — copied verbatim, not re-derived. */
    public static BoardOrder getBoardOrder() {
        int[][] literal = {
                {196, 197, 198, 199, 200, 205, 210, 215, 220, 225, 230, 235, 243, 249, 254, 255},
                {191, 192, 193, 194, 195, 204, 209, 214, 219, 224, 229, 234, 242, 248, 252, 253},
                {186, 187, 188, 189, 190, 203, 208, 213, 218, 223, 228, 233, 241, 247, 250, 251},
                {181, 182, 183, 184, 185, 202, 207, 212, 217, 222, 227, 232, 240, 244, 245, 246},
                {176, 177, 178, 179, 180, 201, 206, 211, 216, 221, 226, 231, 236, 237, 238, 239},
                {160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175},
                {144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159},
                {128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143},
                {112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127},
                {96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111},
                {80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95},
                {64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79},
                {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
                {32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47},
                {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
        };
        int[] rows = new int[256];
        int[] cols = new int[256];
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 16; col++) {
                int step = literal[15 - row][col];
                rows[step] = row;
                cols[step] = col;
            }
        }
        return new BoardOrder(rows, cols);
    }

    public static List<BwPiece> getPieces(String piecesFilePath) throws IOException {
        int[][] raw;
        try {
            raw = JBlackwoodToBucas.loadHisPieces(piecesFilePath);
        } catch (IOException e) {
            throw new IOException("Could not load " + piecesFilePath
                    + " — this port does not fall back to mock data; are you running from the project root?", e);
        }
        List<BwPiece> pieces = new ArrayList<>(256);
        for (int i = 0; i < 256; i++) {
            int[] r = raw[i]; // {N,E,S,W}
            pieces.add(new BwPiece(i + 1, r[0], r[1], r[2], r[3]));
        }
        return pieces;
    }

    /** Pure (no I/O) — grid text + bucas URL, exactly matching Save_Board's format. */
    public static String buildBoardString(BwRotatedPiece[] board, BwPiece[] pieceByNumber) {
        StringBuilder grid = new StringBuilder();
        StringBuilder url = new StringBuilder();
        for (int i = 15; i >= 0; i--) {
            StringBuilder row = new StringBuilder();
            for (int j = 0; j < 16; j++) {
                BwRotatedPiece cell = board[i * 16 + j];
                if (cell.pieceNumber() > 0) {
                    row.append(String.format("%3d/%d ", cell.pieceNumber(), cell.rotations()));
                    BwPiece p = pieceByNumber[cell.pieceNumber()];
                    switch (cell.rotations()) {
                        case 0 -> appendUrl(url, p.topSide(), p.rightSide(), p.bottomSide(), p.leftSide());
                        case 1 -> appendUrl(url, p.leftSide(), p.topSide(), p.rightSide(), p.bottomSide());
                        case 2 -> appendUrl(url, p.bottomSide(), p.leftSide(), p.topSide(), p.rightSide());
                        case 3 -> appendUrl(url, p.rightSide(), p.bottomSide(), p.leftSide(), p.topSide());
                        default -> throw new IllegalStateException("Unexpected rotation: " + cell.rotations());
                    }
                } else {
                    row.append("---/- ");
                    url.append("aaaa");
                }
            }
            grid.append(row).append('\n');
        }
        return grid + "\n" + "https://e2.bucas.name/#puzzle=Knud_Hansen&board_w=16&board_h=16&board_edges="
                + url + "&motifs_order=jblackwood";
    }

    private static void appendUrl(StringBuilder url, int a, int b, int c, int d) {
        url.append((char) (a + 'a')).append((char) (b + 'a')).append((char) (c + 'a')).append((char) (d + 'a'));
    }

    /** I/O wrapper around buildBoardString — fresh Random per call, matches Save_Board. */
    public static Path saveBoard(BwRotatedPiece[] board, int maxSolveIndex, BwPiece[] pieceByNumber, Path outputDir)
            throws IOException, NoSuchAlgorithmException {
        String boardString = buildBoardString(board, pieceByNumber);
        byte[] md5 = MessageDigest.getInstance("MD5").digest(boardString.getBytes(StandardCharsets.UTF_8));
        StringBuilder hex = new StringBuilder();
        for (byte b : md5) {
            hex.append(String.format("%02x", b));
        }
        String filename = maxSolveIndex + "_" + hex + "_" + new Random().nextInt(1000000) + ".txt";
        Files.createDirectories(outputDir);
        Path file = outputDir.resolve(filename);
        Files.writeString(file, boardString, StandardCharsets.UTF_8);
        return file;
    }
}
