package dk.puzzle.blackwood;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link BwUtil}, the table-construction/algorithm-data
 * helpers of the faithful Blackwood port. All assertions are cross-checked
 * directly against Blackwood's C# source ({@code Util.cs}/{@code Program.cs})
 * rather than hand-derived, per this port's phase-1 fidelity goal.
 */
class BwUtilTest {

    @Test
    void testCalculateTwoSides() {
        assertEquals(0, BwUtil.calculateTwoSides(0, 0));
        assertEquals(528, BwUtil.calculateTwoSides(22, 22));
        assertEquals(23, BwUtil.calculateTwoSides(1, 0));
    }

    @Test
    void testGetBoardOrderInvariantsHoldForEveryStep() {
        BwUtil.BoardOrder order = BwUtil.getBoardOrder();

        // step[row][col] = the sequence index at which (row,col) is filled.
        int[][] stepOf = new int[16][16];
        for (int step = 0; step < 256; step++) {
            stepOf[order.rows()[step]][order.cols()[step]] = step;
        }

        for (int step = 1; step < 256; step++) {
            int row = order.rows()[step];
            int col = order.cols()[step];
            if (col > 0) {
                assertTrue(stepOf[row][col - 1] < step,
                        "west neighbour of step " + step + " (" + row + "," + col + ") must already be placed");
            }
            if (row > 0) {
                assertTrue(stepOf[row - 1][col] < step,
                        "south neighbour of step " + step + " (" + row + "," + col + ") must already be placed");
            }
        }
    }

    @Test
    void testGetBoardOrderSpotAnchors() {
        BwUtil.BoardOrder order = BwUtil.getBoardOrder();

        assertEquals(0, order.rows()[0]);
        assertEquals(0, order.cols()[0]); // step 0 = the seeded bottom-left corner

        assertEquals(15, order.rows()[255]);
        assertEquals(15, order.cols()[255]); // step 255 = the last cell, opposite corner

        assertEquals(1, order.rows()[16]);
        assertEquals(0, order.cols()[16]);

        assertEquals(10, order.rows()[175]);
        assertEquals(15, order.cols()[175]);
    }

    @Test
    void testGetRotatedPiecesRotationZeroForPieceOne() {
        // Real piece #1 (a corner): Top=1, Right=17, Bottom=0, Left=0.
        BwPiece piece1 = new BwPiece(1, 1, 17, 0, 0);

        List<BwUtil.RotatedCandidate> candidates = BwUtil.getRotatedPieces(piece1, false);
        BwUtil.RotatedCandidate rot0AtOrigin = candidates.stream()
                .filter(c -> c.leftBottom() == BwUtil.calculateTwoSides(0, 0) && c.rotatedPiece().rotations() == 0)
                .findFirst().orElseThrow();

        assertEquals(0, rot0AtOrigin.rotatedPiece().breakCount());
        assertEquals(1, rot0AtOrigin.rotatedPiece().topSide());
        assertEquals(17, rot0AtOrigin.rotatedPiece().rightSide());
        assertEquals(0, rot0AtOrigin.rotatedPiece().heuristicSideCount()); // none of {1,17,0,0} are in {13,16,10}
    }

    @Test
    void testGetRotatedPiecesDisqualifiesSideEdgeMismatchEvenWithBreaksAllowed() {
        // Synthetic piece whose TopSide (1) is a SIDE_EDGES colour. At (left=0,bottom=4):
        // rotation 3 checks (TopSide vs left) and (LeftSide vs bottom) -- TopSide(1)!=0 is the
        // only mismatch, LeftSide(4)==bottom(4) matches -- so breakCount=1 but the mismatching
        // edge (TopSide=1) is itself a side-only colour, so it must be excluded even with
        // allowBreaks=true, not merely penalized.
        BwPiece piece = new BwPiece(999, 1, 2, 3, 4);
        int leftBottom = BwUtil.calculateTwoSides(0, 4);

        List<BwUtil.RotatedCandidate> withBreaks = BwUtil.getRotatedPieces(piece, true);
        boolean anyRotation3AtKey = withBreaks.stream()
                .anyMatch(c -> c.leftBottom() == leftBottom && c.rotatedPiece().rotations() == 3);

        assertFalse(anyRotation3AtKey, "a mismatch on a SIDE_EDGES-coloured edge must never be offered as a break candidate");
    }

    @Test
    void testGetRotatedPiecesAllowBreaksFalseIsSubsetOfTrue() {
        // Colours deliberately avoid both SIDE_EDGES {1,5,9,13,17} and HEURISTIC_SIDES {13,16,10}
        // so this test isolates the break-tolerance behaviour alone.
        BwPiece piece = new BwPiece(1000, 2, 3, 4, 6);

        List<BwUtil.RotatedCandidate> noBreaks = BwUtil.getRotatedPieces(piece, false);
        List<BwUtil.RotatedCandidate> withBreaks = BwUtil.getRotatedPieces(piece, true);

        assertTrue(withBreaks.size() > noBreaks.size(), "allowing breaks must strictly add candidates for a piece with no side-edge colours");
        assertTrue(withBreaks.containsAll(noBreaks), "every zero-break candidate must still be present when breaks are allowed");
        assertTrue(noBreaks.stream().allMatch(c -> c.rotatedPiece().breakCount() == 0));
        assertTrue(withBreaks.stream().anyMatch(c -> c.rotatedPiece().breakCount() == 1));
    }

    @Test
    void testSortAndFreezeByScoreOrdersDescendingAndDropsScore() {
        // Both pieces share Left=2,Bottom=13 so their rotation-0 candidates collide at the same
        // table key (2,13), letting this test observe them ordered against each other directly.
        BwPiece high = new BwPiece(1, 10, 16, 13, 2); // Top,Right,Bottom all heuristic-coloured -> score 300
        BwPiece low = new BwPiece(2, 3, 4, 13, 2); // only Bottom heuristic-coloured -> score 100

        int key = BwUtil.calculateTwoSides(2, 13);

        List<BwUtil.RotatedCandidate> all = new java.util.ArrayList<>();
        all.addAll(BwUtil.getRotatedPieces(high, false));
        all.addAll(BwUtil.getRotatedPieces(low, false));

        Map<Integer, List<BwUtil.RotatedCandidate>> grouped = BwUtil.groupByLeftBottom(all);
        // A fixed seed is fine here: the invariant under test (score gap of 200 exceeds the max
        // possible jitter spread of 99) holds for ANY jitter draw, deterministic or not.
        Random rand = new Random(42);
        BwRotatedPiece[][] frozen = BwUtil.sortAndFreezeByScore(grouped, rand);

        BwRotatedPiece[] atKey = frozen[key];
        assertNotNull(atKey, "expected both pieces' rotation-0 candidates to land at key " + key);
        List<Integer> pieceNumbersAtKey = java.util.Arrays.stream(atKey).map(BwRotatedPiece::pieceNumber).toList();
        assertTrue(pieceNumbersAtKey.contains(1) && pieceNumbersAtKey.contains(2),
                "expected both piece 1 and piece 2 at key " + key + ", got " + pieceNumbersAtKey);
        assertTrue(pieceNumbersAtKey.indexOf(1) < pieceNumbersAtKey.indexOf(2),
                "piece 1 (score 300) must always sort before piece 2 (score 100) -- gap (200) exceeds max jitter (99): " + pieceNumbersAtKey);
    }

    @Test
    void testFirstBreakIndexAndBreakArray() {
        // firstBreakIndex() must stay 201 regardless of HINT_BREAK_INDEXES -- it drives
        // middlesNoBreak/middlesWithBreak selection for every ordinary (non-hint) cell, and must
        // not be dragged down to 34 by the hint-specific early break points.
        assertEquals(201, BwUtil.firstBreakIndex());

        int[] arr = BwUtil.getBreakArray();
        // 2026-09-04: HINT_BREAK_INDEXES {34, 45, 188, 247} now has one entry per non-center
        // hint's own fill-step (34=hint181, 45=hint249, 188=hint208, 247=hint255 -- see the
        // corrected write-up on HINT_BREAK_INDEXES for how this was verified), each adding 1 to
        // the cumulative budget exactly when that hint is reached. Values below confirmed by
        // actually running getBreakArray(), not hand-derived.
        assertEquals(0, arr[33]);
        assertEquals(1, arr[34]);
        assertEquals(1, arr[44]);
        assertEquals(2, arr[45]);
        assertEquals(2, arr[187]);
        assertEquals(3, arr[188]);
        assertEquals(3, arr[200]);
        assertEquals(4, arr[201]);
        assertEquals(5, arr[206]);
        assertEquals(6, arr[211]);
        assertEquals(7, arr[216]);
        assertEquals(8, arr[221]);
        assertEquals(9, arr[225]);
        assertEquals(10, arr[229]);
        assertEquals(11, arr[233]);
        assertEquals(12, arr[237]);
        assertEquals(12, arr[238]);
        // Tracks BwUtil.BREAK_INDEXES_ALLOWED -- update whenever that schedule changes.
        // Blackwood's original 10-break schedule: the 10th break unlocks at 239 and the budget
        // then tops out at 10 (now 14, +4 for the hint breaks -- one per hint). (239 was briefly
        // dropped 2026-08-24, restored 2026-08-30 -- see the write-up on BREAK_INDEXES_ALLOWED.)
        assertEquals(13, arr[239]);
        assertEquals(13, arr[246]);
        assertEquals(14, arr[247]);
        assertEquals(14, arr[255]);
    }

    @Test
    void testGetHeuristicArrayMatchesIndependentReimplementation() {
        // Reimplemented independently (not copy-pasted from BwUtil) using the exact same
        // literal branches as Program.cs, including its float/double precision asymmetry,
        // so this test can't just be "the same bug twice". Boundary values here are
        // float32-rounding-sensitive -- do not replace this with hardcoded expected values.
        int[] expected = new int[256];
        for (int i = 0; i < 256; i++) {
            if (i <= 16) {
                expected[i] = 0;
            } else if (i <= 26) {
                expected[i] = (int) (((float) i - 16) * 2.8f);
            } else if (i <= 56) {
                expected[i] = (int) ((((float) i - 26) * 1.43333f) + 28);
            } else if (i <= 76) {
                expected[i] = (int) ((((float) i - 56) * 0.9f) + 71);
            } else if (i <= 102) {
                expected[i] = (int) ((((float) i - 76) * 0.6538f) + 89);
            } else if (i <= 160) {
                expected[i] = (int) ((((float) i - 102) / 4.4615) + 106);
            }
        }

        assertArrayEquals(expected, BwUtil.getHeuristicArray());
    }

    @Test
    void testGetHeuristicArraySafeInvariants() {
        int[] arr = BwUtil.getHeuristicArray();
        for (int i = 0; i <= 16; i++) {
            assertEquals(0, arr[i], "index " + i);
        }
        for (int i = 161; i < 256; i++) {
            assertEquals(0, arr[i], "index " + i + " is never read by the solver and defaults to zero");
        }
    }

    @Test
    void testGetPiecesLoadsRealResourceAndMatchesKnownValues() throws Exception {
        List<BwPiece> pieces = BwUtil.getPieces("src/main/resources/JBlackwood_Pieces.txt");

        assertEquals(256, pieces.size());

        BwPiece one = pieces.get(0);
        assertEquals(1, one.pieceNumber());
        assertEquals(1, one.topSide());
        assertEquals(17, one.rightSide());
        assertEquals(0, one.bottomSide());
        assertEquals(0, one.leftSide());
        assertEquals(2, one.pieceType());

        BwPiece p139 = pieces.get(138);
        assertEquals(139, p139.pieceNumber());
        assertEquals(6, p139.topSide());
        assertEquals(11, p139.rightSide());
        assertEquals(18, p139.bottomSide());
        assertEquals(6, p139.leftSide());
        assertEquals(0, p139.pieceType());

        BwPiece p256 = pieces.get(255);
        assertEquals(256, p256.pieceNumber());
        assertEquals(21, p256.topSide());
        assertEquals(22, p256.rightSide());
        assertEquals(8, p256.bottomSide());
        assertEquals(22, p256.leftSide());
    }

    @Test
    void testGetPiecesFailsLoudOnMissingFile() {
        assertThrows(java.io.IOException.class, () -> BwUtil.getPieces("does/not/exist.txt"));
    }

    @Test
    void testBuildBoardStringGridAndUrlFormat() {
        BwRotatedPiece[] board = new BwRotatedPiece[256];
        java.util.Arrays.fill(board, BwRotatedPiece.EMPTY);
        BwPiece[] byNumber = new BwPiece[257];

        // Synthetic piece with 4 distinguishable colours: N=1,E=2,S=3,W=4.
        byNumber[1] = new BwPiece(1, 1, 2, 3, 4);
        // Place it at physical row 0, col 0 (bottom-left, printed last) with rotation 1.
        board[0] = new BwRotatedPiece(1, 1, 4 /*topSide after rot1 = canonical Left*/, 1 /*rightSide after rot1 = canonical Top*/, 0, 0);

        String result = BwUtil.buildBoardString(board, byNumber);
        String[] parts = result.split("\n\nhttps://");
        String[] rows = parts[0].split("\n");

        assertEquals(16, rows.length);
        assertEquals("---/- ".repeat(16).trim(), rows[0].trim(), "top printed row (physical row 15) has no pieces");
        assertTrue(rows[15].startsWith("  1/1 "), "bottom printed row (physical row 0) shows piece 1 rotation 1: " + rows[15]);

        String url = "https://" + parts[1];
        assertTrue(url.startsWith("https://e2.bucas.name/#puzzle=Knud_Hansen&board_w=16&board_h=16&board_edges="));
        assertTrue(url.endsWith("&motifs_order=jblackwood"));

        // board_edges is 4 chars per cell, appended in the same i=15..0,j=0..15 order the grid
        // rows are printed in. Cell (row=0,col=0) is processed at position k=(15-0)*16+0=240 of
        // 256 -- i.e. it is NOT the last cell (that's row=0,col=15) -- preceded by 240 "aaaa"
        // empties and followed by 15 more.
        String edgesSection = url.substring(url.indexOf("board_edges=") + "board_edges=".length(), url.indexOf("&motifs_order"));
        assertEquals(1024, edgesSection.length());
        String cell0Edges = edgesSection.substring(240 * 4, 240 * 4 + 4);
        // rotation 1 output order is Left,Top,Right,Bottom = 4,1,2,3 -> chars 'e','b','c','d'
        assertEquals("ebcd", cell0Edges);
        assertEquals("aaaa".repeat(240), edgesSection.substring(0, 240 * 4));
        assertEquals("aaaa".repeat(15), edgesSection.substring(240 * 4 + 4));
    }
}
