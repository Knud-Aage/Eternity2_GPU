package dk.puzzle.blackwood;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Covers {@link BwGpuTables#buildVariants}, added 2026-08-31 so the GPU kernel can stop having all
 * 16384 threads share one frozen candidate ordering for an entire epoch. The property that matters
 * most here isn't that variants differ (trivially true) -- it's that every variant, individually,
 * still satisfies the break-count-then-heuristic-count monotonicity the kernel's own early-exit
 * logic depends on (see SolveBlackwoodKernel.cu's "sort-order invariant" comment at the main
 * candidate scan). A variant that violated this wouldn't crash; it would silently make the search
 * give up on valid branches early. That is checked directly here, not just reasoned about.
 */
class BwGpuTablesVariantsTest {

    private static final String PIECES_PATH = "src/main/resources/JBlackwood_Pieces.txt";

    private static BlackwoodSolver freshSolver() throws Exception {
        BlackwoodSolver solver = new BlackwoodSolver(190, Path.of("build", "test-output"), 1, PIECES_PATH);
        solver.prepare();
        return solver;
    }

    private static int breakCountOf(int rec) {
        return (rec >> 26) & 0x1;
    }

    private static int heuristicCountOf(int rec) {
        return (rec >> 27) & 0x7;
    }

    @Test
    void producesExactlyNVariantsOfEqualLength() throws Exception {
        BwGpuTables.MultiVariantTableSet set = BwGpuTables.buildVariants(freshSolver(), 8);
        assertEquals(8, set.payloadVariants().length);
        int expectedLength = set.payloadVariants()[0].length;
        for (int[] variant : set.payloadVariants()) {
            assertEquals(expectedLength, variant.length);
        }
    }

    @Test
    void csrStructureMatchesAPlainSingleVariantBuild() throws Exception {
        BlackwoodSolver solver = freshSolver();
        BwGpuTables.GpuTableSet single = BwGpuTables.build(solver);
        BwGpuTables.MultiVariantTableSet multi = BwGpuTables.buildVariants(solver, 4);
        // buildVariants re-prepare()s internally, but bucket structure must not depend on that.
        assertArrayEquals(single.csrCount(), multi.csrCount());
        assertEquals(single.payload().length, multi.payloadVariants()[0].length);
    }

    @Test
    void atLeastOneVariantGenuinelyDiffersFromTheFirst() throws Exception {
        // Each call to prepare() draws a fresh, unseeded Random, so with thousands of entries the
        // chance every single one of several variants coincidentally matches variant 0 exactly is
        // astronomically small. If this ever flakes, the diversity mechanism itself is broken.
        BwGpuTables.MultiVariantTableSet set = BwGpuTables.buildVariants(freshSolver(), 8);
        boolean anyDiffers = false;
        for (int v = 1; v < set.payloadVariants().length; v++) {
            if (!Arrays.equals(set.payloadVariants()[0], set.payloadVariants()[v])) {
                anyDiffers = true;
                break;
            }
        }
        assertTrue(anyDiffers, "all 8 variants were byte-for-byte identical -- prepare() is not re-randomizing");
    }

    @Test
    void everyVariantPreservesBreakCountThenHeuristicCountMonotonicityPerBucket() throws Exception {
        BwGpuTables.MultiVariantTableSet set = BwGpuTables.buildVariants(freshSolver(), 6);
        int[] csrOffset = set.csrOffset();
        int[] csrCount = set.csrCount();

        for (int[] payload : set.payloadVariants()) {
            for (int bucket = 0; bucket < csrOffset.length; bucket++) {
                int off = csrOffset[bucket];
                int cnt = csrCount[bucket];
                if (cnt <= 1) continue;

                for (int i = 1; i < cnt; i++) {
                    int prev = payload[off + i - 1];
                    int cur = payload[off + i];
                    int prevBreak = breakCountOf(prev);
                    int curBreak = breakCountOf(cur);
                    // Primary key: break count must never decrease as we scan forward.
                    assertTrue(curBreak >= prevBreak,
                            "bucket " + bucket + " index " + i + ": breakCount decreased (" + prevBreak + " -> " + curBreak + ")");
                    // Secondary key, only meaningful within an unchanged break-count tier: heuristic
                    // count must never increase (the kernel's heuristic-floor early-exit relies on
                    // "if this one fails, no later one in this tier could pass either").
                    if (curBreak == prevBreak) {
                        int prevHeuristic = heuristicCountOf(prev);
                        int curHeuristic = heuristicCountOf(cur);
                        assertTrue(curHeuristic <= prevHeuristic,
                                "bucket " + bucket + " index " + i + ": heuristicCount rose within a break-count tier ("
                                        + prevHeuristic + " -> " + curHeuristic + ")");
                    }
                }
            }
        }
    }

    @Test
    void rejectsFewerThanOneVariant() throws Exception {
        assertThrows(IllegalArgumentException.class, () -> BwGpuTables.buildVariants(freshSolver(), 0));
    }
}
