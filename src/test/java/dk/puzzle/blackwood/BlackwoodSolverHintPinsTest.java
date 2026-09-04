package dk.puzzle.blackwood;

import org.junit.jupiter.api.AfterAll;
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
 *
 * <p>2026-09-02: the 4 non-center hints are now behind {@code BlackwoodSolver.NON_CENTER_HINTS_ENABLED},
 * off by default -- explicitly turned on here for the duration of this test class (and restored
 * after) since these tests are specifically about that feature. See
 * {@code BlackwoodSolverNoHintsTest} for coverage of the default (off) state.
 *
 * <p>2026-09-04: all four non-center hints are now break-tolerant, not just hint208/hint255 --
 * see the corrected write-up on {@code BwUtil.HINT_BREAK_INDEXES} for why hint181/hint249 turned
 * out to need this more, not less, than 208/255 did.
 */
class BlackwoodSolverHintPinsTest {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";
    private static BlackwoodSolver solver;

    @BeforeAll
    static void prepareSolver() throws Exception {
        BlackwoodSolver.NON_CENTER_HINTS_ENABLED = true;
        solver = new BlackwoodSolver(190, Path.of("build", "test-output"), 1, PIECES_PATH);
        solver.prepare();
    }

    @AfterAll
    static void restoreDefault() {
        BlackwoodSolver.NON_CENTER_HINTS_ENABLED = false;
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

    /**
     * For all four non-center hints (allowBreaks=true, 2026-09-02 for 208/255, extended
     * 2026-09-04 to 181/249): every bucket must still hold only the pinned piece+rotation, in
     * exactly one candidate each, but there should be MANY buckets now (one exact break-free
     * match plus ~44 one-side-tolerant break variants) instead of just one -- otherwise the
     * allowBreaks change silently did nothing.
     */
    private static void assertBreakTolerantPin(BwRotatedPiece[][] table, int expectedPiece, int expectedRotation) {
        assertNotNull(table, "table must exist");
        int nonEmptyBuckets = 0;
        boolean sawExactMatch = false;
        for (BwRotatedPiece[] bucket : table) {
            if (bucket == null || bucket.length == 0) continue;
            nonEmptyBuckets++;
            assertEquals(1, bucket.length, "each bucket must still have exactly one candidate");
            BwRotatedPiece found = bucket[0];
            assertEquals(expectedPiece, found.pieceNumber(), "wrong piece pinned");
            assertEquals(expectedRotation, found.rotations(), "wrong rotation for piece " + expectedPiece);
            if (found.breakCount() == 0) sawExactMatch = true;
        }
        assertTrue(nonEmptyBuckets > 1,
                "expected many break-tolerant buckets for piece " + expectedPiece + ", found only " + nonEmptyBuckets);
        assertTrue(sawExactMatch, "the break-free exact match must still be present for piece " + expectedPiece);
    }

    @Test
    void centerPinIsPiece139AtRotation2() {
        // The mandatory center pin predates the whole hint feature and has never shown the
        // near-universal-stall problem the other four did -- stays a hard pin.
        assertPin(solver.start, 139, 2);
    }

    @Test
    void hint208PinIsBreakTolerant() {
        assertBreakTolerantPin(solver.hint208, 208, 2);
    }

    @Test
    void hint255PinIsBreakTolerant() {
        assertBreakTolerantPin(solver.hint255, 255, 2);
    }

    @Test
    void hint181PinIsBreakTolerant() {
        // 2026-09-04: this is the hint actually reached at fill-step 34 (verified against the
        // live masterPieceLookup) -- the earlier {34, 45} comment describing this as hint208's
        // territory was wrong; see BwUtil.HINT_BREAK_INDEXES for the correction.
        assertBreakTolerantPin(solver.hint181, 181, 2);
    }

    @Test
    void hint249PinIsBreakTolerant() {
        assertBreakTolerantPin(solver.hint249, 249, 3);
    }

    @Test
    void hintBreakIndexesDoNotDisturbGeneralFirstBreakIndex() {
        assertEquals(201, BwUtil.firstBreakIndex(),
                "adding hint-specific break points must not move the general break-schedule cutoff");
    }

    @Test
    void hintBreakBudgetStacksCorrectlyWithGeneralSchedule() {
        // 2026-09-04: HINT_BREAK_INDEXES is now {34, 45, 188, 247} -- one entry per hint's own
        // fill-step (181@34, 249@45, 208@188, 255@247), each contributing its own +1 so no
        // earlier hint's break usage can starve a later one out of its own budget. See
        // BwUtilTest for the fuller checkpoint list; this test just pins down the four unlock
        // points themselves.
        int[] breakArray = BwUtil.getBreakArray();
        assertEquals(1, breakArray[34], "hint181's own break budget should unlock at its own step, 34");
        assertEquals(2, breakArray[45], "hint249's own break budget should unlock at its own step, 45");
        assertEquals(2, breakArray[187], "budget must stay at 2 right up until hint208's own step");
        assertEquals(3, breakArray[188], "hint208's own break budget should unlock at its own step, 188");
        assertEquals(3, breakArray[200], "budget must stay at 3 right up to the general schedule");
        assertEquals(4, breakArray[201], "general schedule's first entry (201) should add on top");
        assertEquals(13, breakArray[246], "budget must stay at 13 right up until hint255's own step");
        assertEquals(14, breakArray[247], "hint255's own break budget should unlock at its own step, 247");
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
