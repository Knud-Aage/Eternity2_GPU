package dk.puzzle.blackwood;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Confirms BwGpuTables -- the data actually shipped to the CUDA kernel -- correctly reflects
 * BlackwoodSolver.ROTATE_INSTANCE_DEGREES, not just the CPU-only solvePuzzle() path that
 * BlackwoodSolverRotationTest and the standalone benchmark exercised. BwGpuTables.build() works
 * entirely off reference-identity against solver's own fields (masterPieceLookup, hint181, ...),
 * so this is expected to work "by construction" -- this test makes that a verified fact instead
 * of an assumption, on the exact code path BlackwoodGpuRunner uses every epoch.
 */
class BwGpuTablesRotationTest {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";

    private static BlackwoodSolver solver;
    private static BwGpuTables.GpuTableSet tables;

    @BeforeAll
    static void prepareRotatedSolverAndTables() throws Exception {
        BlackwoodSolver.NON_CENTER_HINTS_ENABLED = true;
        BlackwoodSolver.ROTATE_INSTANCE_DEGREES = 180;
        solver = new BlackwoodSolver(190, Path.of("build", "test-output"), 1, PIECES_PATH);
        solver.prepare();
        tables = BwGpuTables.build(solver);
    }

    @AfterAll
    static void restoreDefaults() {
        BlackwoodSolver.ROTATE_INSTANCE_DEGREES = 0;
        BlackwoodSolver.NON_CENTER_HINTS_ENABLED = false;
    }

    /** Same order BwGpuTables.tablesInOrder() uses -- independent here so this test can't share a bug with production code. */
    private static BwRotatedPiece[][][] expectedTablesInOrder() {
        return new BwRotatedPiece[][][]{
                solver.corners, solver.leftSides, solver.topSides,
                solver.rightSidesWithoutBreaks, solver.rightSidesWithBreaks,
                solver.middlesNoBreak, solver.middlesWithBreak,
                solver.southStart, solver.westStart, solver.start,
                solver.hint208, solver.hint255, solver.hint181, solver.hint249
        };
    }

    @Test
    void stepToTableIdStillReferenceMatchesMasterPieceLookupUnderRotation() {
        BwRotatedPiece[][][] expected = expectedTablesInOrder();
        for (int step = 0; step < 256; step++) {
            int row = solver.boardOrderRow[step];
            int col = solver.boardOrderCol[step];
            if (row == 0) continue;

            BwRotatedPiece[][] expectedTable = solver.masterPieceLookup[row * 16 + col];
            int tableId = tables.stepToTableId()[step];
            assertSame(expectedTable, expected[tableId],
                    "step=" + step + " (row=" + row + ",col=" + col + ") table id " + tableId
                            + " doesn't reference-match masterPieceLookup under 180-degree rotation");
        }
    }

    /**
     * Hand-derived from BlackwoodSolverRotationTest's already-verified 180-degree table: at
     * 180 degrees, hint181 sits at (13,13), hint249 at (13,2), hint208 at (2,13), hint255 at
     * (2,2), center stays at (7,7). This pins down that BwGpuTables genuinely moved -- not just
     * that it's internally self-consistent -- by checking the GPU step-to-table mapping sends
     * the rotated cells' fill-steps to the expected table, by reference identity.
     */
    @Test
    void gpuTablesActuallyReflectTheRotatedHintCellsNotJustTheUnrotatedOnes() {
        record Expected(int row, int col, BwRotatedPiece[][] table) {
        }
        Expected[] expected = {
                new Expected(7, 7, solver.start),
                new Expected(13, 13, solver.hint181),
                new Expected(13, 2, solver.hint249),
                new Expected(2, 13, solver.hint208),
                new Expected(2, 2, solver.hint255),
        };
        BwRotatedPiece[][][] tablesInOrder = expectedTablesInOrder();
        for (Expected e : expected) {
            int step = -1;
            for (int s = 0; s < 256; s++) {
                if (solver.boardOrderRow[s] == e.row() && solver.boardOrderCol[s] == e.col()) {
                    step = s;
                    break;
                }
            }
            assertTrue(step >= 0, "no fill-step found for (" + e.row() + "," + e.col() + ")");
            int tableId = tables.stepToTableId()[step];
            assertSame(e.table(), tablesInOrder[tableId],
                    "cell (" + e.row() + "," + e.col() + ") at fill-step " + step
                            + " should route to its rotated hint's table in the GPU step-to-table mapping");
        }
    }

    @Test
    void csrPayloadForHint181TableStillRoundTripsUnderRotation() {
        BwRotatedPiece[][] expectedEntries2d = solver.hint181;
        int tableId = -1;
        BwRotatedPiece[][][] tablesInOrder = expectedTablesInOrder();
        for (int t = 0; t < tablesInOrder.length; t++) {
            if (tablesInOrder[t] == expectedEntries2d) {
                tableId = t;
                break;
            }
        }
        assertTrue(tableId >= 0, "hint181 table must be one of the known GPU tables");

        int nonEmptyKeys = 0;
        for (int key = 0; key < BwGpuTables.KEY_SPACE; key++) {
            int idx = tableId * BwGpuTables.KEY_SPACE + key;
            int count = tables.csrCount()[idx];
            if (count == 0) continue;
            nonEmptyKeys++;
            assertEquals(expectedEntries2d[key].length, count, "key " + key + " count mismatch for hint181's table");
        }
        assertTrue(nonEmptyKeys > 0, "hint181's rotated table must still have candidates flattened into the CSR payload");
    }
}
