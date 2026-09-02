package dk.puzzle.blackwood;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Covers the 5 official Eternity II clue pins added 2026-09-01. Position and rotation for the 4
 * non-center pieces were independently re-derived (not taken from any public writeup) and
 * cross-checked three ways -- see the javadoc on {@code BlackwoodSolver.HINT_PINS}. This test
 * exists because that derivation is exactly the kind of thing that can be silently wrong: it
 * asserts each pin resolves to a real, single, correctly-rotated candidate through the solver's
 * OWN matching logic, not just that the wiring compiles and nothing throws.
 */
class BlackwoodSolverHintPinsTest {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static BlackwoodSolver solver;

    @BeforeAll
    static void prepareSolver() throws Exception {
        solver = new BlackwoodSolver(190, Path.of("build", "test-output"), 1, PIECES_PATH);
        solver.prepare();
    }

    private static void assertPin(BwRotatedPiece[][] table, int expectedPiece, int expectedRotation) {
        assertNotNull(table, "table must exist");
        // Single-entry pieces are stored at LeftBottom = whatever candidate scoring computed. Since
        // there is exactly one legal candidate, scan the whole KEY_SPACE for the one non-empty
        // bucket rather than assuming its key.
        BwRotatedPiece found = null;
        int nonEmptyBuckets = 0;
        for (BwRotatedPiece[] bucket : table) {
            if (bucket == null || bucket.length == 0) continue;
            nonEmptyBuckets++;
            assertEquals(1, bucket.length, "a pinned piece must have exactly one candidate in its bucket");
            found = bucket[0];
        }
        assertEquals(1, nonEmptyBuckets, "a pinned piece must occupy exactly one LeftBottom bucket");
        assertNotNull(found, "pin must resolve to a real candidate, not an empty table");
        assertEquals(expectedPiece, found.pieceNumber(), "wrong piece pinned");
        assertEquals(expectedRotation, found.rotations(), "wrong rotation for piece " + expectedPiece);
    }

    @Test
    void centerPinIsPiece139AtRotation2() {
        assertPin(solver.start, 139, 2);
    }

    @Test
    void hint208PinIsCorrect() {
        assertPin(solver.hint208, 208, 2);
    }

    @Test
    void hint255PinIsCorrect() {
        assertPin(solver.hint255, 255, 2);
    }

    @Test
    void hint181PinIsCorrect() {
        assertPin(solver.hint181, 181, 2);
    }

    @Test
    void hint249PinIsCorrect() {
        assertPin(solver.hint249, 249, 3);
    }

    @Test
    void allFiveHintPositionsResolveThroughMasterPieceLookup() {
        // (row, col, expected table) for all 5 official clue positions, 0-indexed.
        record Expected(int row, int col, BwRotatedPiece[][] table, int pieceNumber) {
        }
        Expected[] expected = {
                new Expected(7, 7, solver.start, 139),
                new Expected(2, 2, solver.hint181, 181),
                new Expected(2, 13, solver.hint249, 249),
                new Expected(13, 2, solver.hint208, 208),
                new Expected(13, 13, solver.hint255, 255),
        };
        for (Expected e : expected) {
            BwRotatedPiece[][] actual = solver.masterPieceLookup[e.row() * 16 + e.col()];
            assertTrue(actual == e.table(),
                    "masterPieceLookup at (" + e.row() + "," + e.col() + ") should be the piece " + e.pieceNumber()
                            + " table by reference identity (required for BwGpuTables.build() to match it)");
        }
    }

    @Test
    void hintPiecesAreExcludedFromTheGeneralMiddlePool() {
        // If a hint piece leaked into middlesNoBreak/middlesWithBreak, the search could place it
        // somewhere OTHER than its official cell, silently defeating the whole point of pinning it.
        int[] hintPieces = {139, 208, 255, 181, 249};
        for (BwRotatedPiece[] bucket : solver.middlesNoBreak) {
            if (bucket == null) continue;
            for (BwRotatedPiece p : bucket) {
                for (int hp : hintPieces) {
                    assertTrue(p.pieceNumber() != hp,
                            "hint piece " + hp + " must not appear in the general middlesNoBreak pool");
                }
            }
        }
        for (BwRotatedPiece[] bucket : solver.middlesWithBreak) {
            if (bucket == null) continue;
            for (BwRotatedPiece p : bucket) {
                for (int hp : hintPieces) {
                    assertTrue(p.pieceNumber() != hp,
                            "hint piece " + hp + " must not appear in the general middlesWithBreak pool");
                }
            }
        }
    }
}
