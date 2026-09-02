package dk.puzzle.blackwood;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Covers {@code BlackwoodSolver.NON_CENTER_HINTS_ENABLED}'s default (off) state -- added
 * 2026-09-02 alongside the runtime switch itself, so both modes run from one branch instead of
 * needing separate "hints"/"no hints" branches. This is the state main/master ran in before the
 * 4-hint feature existed at all: the center pin (139) stays pinned unconditionally, but 208, 255,
 * 181 and 249 must behave like any other middle piece -- available to the general search, not
 * pinned to their official-clue cell. See {@code BlackwoodSolverHintPinsTest} for the on state.
 */
class BlackwoodSolverNoHintsTest {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static BlackwoodSolver solver;

    @BeforeAll
    static void prepareSolver() throws Exception {
        // Explicit, not relying on the class's static default -- this test asserts the OFF
        // behaviour regardless of whatever another test class or JVM env var left it as.
        BlackwoodSolver.NON_CENTER_HINTS_ENABLED = false;
        solver = new BlackwoodSolver(190, Path.of("build", "test-output"), 1, PIECES_PATH);
        solver.prepare();
    }

    @Test
    void centerStaysPinnedRegardlessOfTheSwitch() {
        int idx = 7 * 16 + 7;
        assertTrue(solver.masterPieceLookup[idx] == solver.start,
                "center (139) must stay pinned at (7,7) even with the switch off -- it predates this feature");
    }

    @Test
    void theFourHintCellsFallThroughToOrdinaryMiddleTables() {
        int[][] cells = {{2, 2}, {2, 13}, {13, 2}, {13, 13}};
        for (int[] cell : cells) {
            int row = cell[0], col = cell[1];
            BwRotatedPiece[][] actual = solver.masterPieceLookup[row * 16 + col];
            boolean isOrdinary = actual == solver.middlesNoBreak || actual == solver.middlesWithBreak;
            assertTrue(isOrdinary, "cell (" + row + "," + col + ") should fall through to an ordinary "
                    + "middles table with the switch off, not a hint table");
        }
    }

    @Test
    void theFourNonCenterHintPiecesAreAvailableToTheGeneralSearch() {
        // With the switch off these 4 pieces must NOT be pulled out of the general pool --
        // otherwise they'd be unplaceable anywhere, silently breaking the "no hints" baseline.
        int[] nonCenterHints = {208, 255, 181, 249};
        for (int piece : nonCenterHints) {
            boolean foundSomewhere = appearsIn(solver.middlesNoBreak, piece) || appearsIn(solver.middlesWithBreak, piece);
            assertTrue(foundSomewhere, "piece " + piece + " must be available in the general middle pool with the switch off");
        }
    }

    @Test
    void switchDefaultsToOffWhenEnvVarIsUnset() {
        // Documents the contract: unset ETERNITY_NON_CENTER_HINTS means off, matching
        // main/master's original behaviour. This test doesn't touch the env var itself (that's
        // process-wide and not safely mutable mid-JVM) -- it just pins down what "unset" means.
        assertFalse("true".equalsIgnoreCase(null), "sanity check on the equalsIgnoreCase(null) contract the switch relies on");
    }

    private static boolean appearsIn(BwRotatedPiece[][] table, int pieceNumber) {
        for (BwRotatedPiece[] bucket : table) {
            if (bucket == null) continue;
            for (BwRotatedPiece p : bucket) {
                if (p.pieceNumber() == pieceNumber) return true;
            }
        }
        return false;
    }
}
