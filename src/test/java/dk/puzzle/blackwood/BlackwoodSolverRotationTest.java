package dk.puzzle.blackwood;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression cover for BlackwoodSolver's ROTATE_INSTANCE_DEGREES experiment, added 2026-09-04
 * (inspired by Igor Pejic's eternity-ii-dfs-solver -- see the javadoc on BASE_HINT_PINS /
 * activeHintPins() for the technique itself and why the center pin is excluded from rotation).
 *
 * <p>The expected (row, col, rotation-index) values below were independently hand-derived from
 * BASE_HINT_PINS by simulating the same quarter-turn geometry two separate ways (direct forward
 * composition, cross-checked against an inverse-consistency argument for the 270-degree case),
 * and the underlying formula was further checked against Igor Pejic's own hints_rot90.txt /
 * hints_rot270.txt reference data before any of this was written -- see the comment on
 * activeHintPins(). They are asserted here, not derived by calling the method under test, so a
 * regression in the transform actually has something independent to fail against.
 */
class BlackwoodSolverRotationTest {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";

    private static BlackwoodSolver solver90;
    private static BlackwoodSolver solver180;
    private static BlackwoodSolver solver270;

    @BeforeAll
    static void prepareSolvers() throws Exception {
        BlackwoodSolver.NON_CENTER_HINTS_ENABLED = true;
        solver90 = prepareAt(90);
        solver180 = prepareAt(180);
        solver270 = prepareAt(270);
    }

    private static BlackwoodSolver prepareAt(int degrees) throws Exception {
        BlackwoodSolver.ROTATE_INSTANCE_DEGREES = degrees;
        BlackwoodSolver s = new BlackwoodSolver(190, Path.of("build", "test-output"), 1, PIECES_PATH);
        s.prepare();
        return s;
    }

    @AfterAll
    static void restoreDefaults() {
        BlackwoodSolver.ROTATE_INSTANCE_DEGREES = 0;
        BlackwoodSolver.NON_CENTER_HINTS_ENABLED = false;
    }

    private record Expected(int row, int col, int rotation) {
    }

    /** table must resolve, through masterPieceLookup, to exactly (row,col), and hold only expectedPiece at expectedRotation. */
    private static void assertPinAt(BlackwoodSolver s, BwRotatedPiece[][] table, Expected e, int expectedPiece) {
        assertTrue(s.masterPieceLookup[e.row() * 16 + e.col()] == table,
                "piece " + expectedPiece + " should resolve through masterPieceLookup at (" + e.row() + "," + e.col() + ")");
        int nonEmptyBuckets = 0;
        for (BwRotatedPiece[] bucket : table) {
            if (bucket == null || bucket.length == 0) continue;
            nonEmptyBuckets++;
            for (BwRotatedPiece p : bucket) {
                assertEquals(expectedPiece, p.pieceNumber(), "wrong piece pinned at rotated cell");
                assertEquals(e.rotation(), p.rotations(),
                        "wrong rotation for piece " + expectedPiece + " after rotating the instance");
            }
        }
        assertTrue(nonEmptyBuckets >= 1, "table for piece " + expectedPiece + " must not be empty");
    }

    @Test
    void rotate90MovesTheFourCornerHintsToTheHandDerivedCells() {
        assertPinAt(solver90, solver90.start, new Expected(7, 7, 2), 139);
        assertPinAt(solver90, solver90.hint181, new Expected(13, 2, 3), 181);
        assertPinAt(solver90, solver90.hint249, new Expected(2, 2, 0), 249);
        assertPinAt(solver90, solver90.hint208, new Expected(13, 13, 3), 208);
        assertPinAt(solver90, solver90.hint255, new Expected(2, 13, 3), 255);
    }

    @Test
    void rotate180MovesTheFourCornerHintsToTheHandDerivedCells() {
        assertPinAt(solver180, solver180.start, new Expected(7, 7, 2), 139);
        assertPinAt(solver180, solver180.hint181, new Expected(13, 13, 0), 181);
        assertPinAt(solver180, solver180.hint249, new Expected(13, 2, 1), 249);
        assertPinAt(solver180, solver180.hint208, new Expected(2, 13, 0), 208);
        assertPinAt(solver180, solver180.hint255, new Expected(2, 2, 0), 255);
    }

    @Test
    void rotate270MovesTheFourCornerHintsToTheHandDerivedCells() {
        assertPinAt(solver270, solver270.start, new Expected(7, 7, 2), 139);
        assertPinAt(solver270, solver270.hint181, new Expected(2, 13, 1), 181);
        assertPinAt(solver270, solver270.hint249, new Expected(13, 13, 2), 249);
        assertPinAt(solver270, solver270.hint208, new Expected(2, 2, 1), 208);
        assertPinAt(solver270, solver270.hint255, new Expected(13, 2, 1), 255);
    }

    @Test
    void theFourCornerHintCellsAreAlwaysAPermutationOfTheBaseFour() {
        Set<Integer> baseCells = Set.of(2 * 16 + 2, 2 * 16 + 13, 13 * 16 + 2, 13 * 16 + 13);
        for (BlackwoodSolver s : new BlackwoodSolver[]{solver90, solver180, solver270}) {
            Set<Integer> rotatedCells = new HashSet<>();
            for (BwRotatedPiece[][] table : new BwRotatedPiece[][][]{s.hint181, s.hint249, s.hint208, s.hint255}) {
                for (int cell = 0; cell < 256; cell++) {
                    if (s.masterPieceLookup[cell] == table) {
                        rotatedCells.add(cell);
                    }
                }
            }
            assertEquals(baseCells, rotatedCells,
                    "rotation must permute the 4 corner-hint cells among themselves, never onto a new cell");
        }
    }

    @Test
    void centerNeverMovesUnderAnyRotation() {
        for (BlackwoodSolver s : new BlackwoodSolver[]{solver90, solver180, solver270}) {
            assertTrue(s.masterPieceLookup[7 * 16 + 7] == s.start,
                    "center pin must stay at (7,7) regardless of ROTATE_INSTANCE_DEGREES");
        }
    }
}
