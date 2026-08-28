package dk.puzzle.blackwood;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * CSR round-trip tests for {@link BwGpuTables}, run before ever touching GPU hardware -- the
 * cheapest, fastest, most important correctness gate for the GPU port (per the approved plan).
 * No CUDA involved; these only verify the host-side data flattening is faithful to what
 * {@link BlackwoodSolver#prepare()} already produced and verified.
 */
class BwGpuTablesTest {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";

    private static BlackwoodSolver solver;
    private static BwGpuTables.GpuTableSet tables;

    @BeforeAll
    static void prepareSharedSolverAndTables() throws Exception {
        solver = new BlackwoodSolver(190, Path.of("build", "test-output"), 1, PIECES_PATH);
        solver.prepare();
        tables = BwGpuTables.build(solver);
    }

    /** Same table order BwGpuTables.tablesInOrder() uses -- kept independent here rather than
     *  reusing that private helper, so this test can't share a bug with the production code. */
    private static BwRotatedPiece[][][] expectedTablesInOrder() {
        return new BwRotatedPiece[][][]{
                solver.corners, solver.leftSides, solver.topSides,
                solver.rightSidesWithoutBreaks, solver.rightSidesWithBreaks,
                solver.middlesNoBreak, solver.middlesWithBreak,
                solver.southStart, solver.westStart, solver.start
        };
    }

    private static int unpackPieceNumber(int rec) {
        return (rec & 0xFF) + 1;
    }

    private static int unpackTopSide(int rec) {
        return (rec >> 8) & 0xFF;
    }

    private static int unpackRightSide(int rec) {
        return (rec >> 16) & 0xFF;
    }

    private static int unpackRotation(int rec) {
        return (rec >> 24) & 0x3;
    }

    private static int unpackBreakCount(int rec) {
        return (rec >> 26) & 0x1;
    }

    private static int unpackHeuristicCount(int rec) {
        return (rec >> 27) & 0x7;
    }

    private static BwRotatedPiece unpack(int rec) {
        return new BwRotatedPiece(unpackPieceNumber(rec), unpackRotation(rec),
                unpackTopSide(rec), unpackRightSide(rec), unpackBreakCount(rec), unpackHeuristicCount(rec));
    }

    @Test
    void testPackUnpackRoundTripsAtFieldBoundaries() {
        int[] pieceNumbers = {1, 256};
        int[] sides = {0, 22};
        int[] rotations = {0, 1, 2, 3};
        int[] breakCounts = {0, 1};
        int[] heuristicCounts = {0, 4};

        for (int pn : pieceNumbers) {
            for (int top : sides) {
                for (int right : sides) {
                    for (int rot : rotations) {
                        for (int bc : breakCounts) {
                            for (int hc : heuristicCounts) {
                                BwRotatedPiece original = new BwRotatedPiece(pn, rot, top, right, bc, hc);
                                int packed = BwGpuTables.pack(original);
                                BwRotatedPiece roundTripped = unpack(packed);
                                assertEquals(original, roundTripped, "round-trip failed for " + original);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void testCsrRoundTripsAllTenTablesExactlyAndInOrder() {
        BwRotatedPiece[][][] expected = expectedTablesInOrder();

        for (int t = 0; t < BwGpuTables.NUM_TABLES; t++) {
            for (int key = 0; key < BwGpuTables.KEY_SPACE; key++) {
                BwRotatedPiece[] expectedEntries = expected[t][key];
                int idx = t * BwGpuTables.KEY_SPACE + key;
                int offset = tables.csrOffset()[idx];
                int count = tables.csrCount()[idx];

                if (expectedEntries == null || expectedEntries.length == 0) {
                    assertEquals(0, count, "table " + t + " key " + key + " should be empty");
                    continue;
                }

                assertEquals(expectedEntries.length, count, "table " + t + " key " + key + " count mismatch");
                for (int i = 0; i < expectedEntries.length; i++) {
                    BwRotatedPiece actual = unpack(tables.payload()[offset + i]);
                    assertEquals(expectedEntries[i], actual,
                            "table " + t + " key " + key + " entry " + i + " mismatch");
                }
            }
        }
    }

    @Test
    void testBottomRawMatchesBottomSidePiecesRotated() {
        for (int left = 0; left < 23; left++) {
            int key = left * 23;
            List<BwUtil.RotatedCandidate> expected = solver.bottomSidePiecesRotated.get(key);
            int offset = tables.bottomRawOffset()[left];
            int count = tables.bottomRawCount()[left];

            if (expected == null || expected.isEmpty()) {
                assertEquals(0, count, "left=" + left + " should be empty");
                continue;
            }

            assertEquals(expected.size(), count, "left=" + left + " count mismatch");
            for (int i = 0; i < expected.size(); i++) {
                BwRotatedPiece actual = unpack(tables.bottomRawPayload()[offset + i]);
                assertEquals(expected.get(i).rotatedPiece(), actual, "left=" + left + " entry " + i + " mismatch");
            }
        }
    }

    @Test
    void testStepToTableIdReferenceMatchesMasterPieceLookup() {
        BwRotatedPiece[][][] expected = expectedTablesInOrder();

        for (int step = 0; step < 256; step++) {
            int row = solver.boardOrderRow[step];
            int col = solver.boardOrderCol[step];

            if (row == 0) {
                assertEquals(BwGpuTables.TABLE_UNUSED_ROW0, tables.stepToTableId()[step],
                        "row 0 steps must be the unused sentinel (handled directly in the kernel), step=" + step);
                continue;
            }

            BwRotatedPiece[][] expectedTable = solver.masterPieceLookup[row * 16 + col];
            int tableId = tables.stepToTableId()[step];
            assertTrue(tableId >= 0 && tableId < BwGpuTables.NUM_TABLES,
                    "step=" + step + " has an out-of-range table id: " + tableId);
            assertSame(expectedTable, expected[tableId],
                    "step=" + step + " (row=" + row + ",col=" + col + ") table id " + tableId + " doesn't reference-match masterPieceLookup");
        }
    }

    @Test
    void testStepBoardIdxMatchesBoardOrder() {
        for (int step = 0; step < 256; step++) {
            int expected = solver.boardOrderRow[step] * 16 + solver.boardOrderCol[step];
            assertEquals(expected, tables.stepBoardIdx()[step], "step=" + step);
        }
    }

    @Test
    void testBreakArrayAndHeuristicArrayPassThroughUnchanged() {
        assertArrayEquals(solver.breakArray, tables.breakArray());
        assertArrayEquals(solver.heuristicArray, tables.heuristicArray());
    }

    @Test
    void testCornersKeyZeroIsNonEmpty() {
        // Step 0's seed pick depends on this being non-empty -- BlackwoodSolver.prepare() already
        // guards this with an IllegalStateException, but confirm it survives the CSR flattening too.
        int idx = BwGpuTables.TABLE_CORNERS * BwGpuTables.KEY_SPACE; // key 0
        assertTrue(tables.csrCount()[idx] > 0, "corners table must have at least one candidate at key 0");
    }

    @Test
    void testPayloadSizesAreWithinPlannedHeadroom() {
        // Measured against real data (2026-08-02): payload is 38,675 (the plan's rough
        // pre-implementation estimate was ~32,600 -- 19% low) and bottomRawPayload is exactly 56
        // (matched the ~56 estimate exactly). BlackwoodGpuEngine.MAX_PAYLOAD_SIZE (50,000, ~29%
        // headroom over the real count) and MAX_BOTTOM_PAYLOAD_SIZE (96) must stay in sync with
        // the thresholds asserted here -- if this test's thresholds ever need to move, that
        // buffer sizing needs to move with them.
        System.out.println("BwGpuTables real payload size: " + tables.payload().length + " (headroom target: 50,000)");
        System.out.println("BwGpuTables real bottomRawPayload size: " + tables.bottomRawPayload().length + " (headroom target: 96)");

        assertTrue(tables.payload().length < 50_000,
                "payload size " + tables.payload().length + " exceeds the planned 50,000 headroom -- re-check BlackwoodGpuEngine's buffer sizing before proceeding");
        assertTrue(tables.bottomRawPayload().length < 96,
                "bottomRawPayload size " + tables.bottomRawPayload().length + " exceeds the planned 96 headroom");
    }
}
