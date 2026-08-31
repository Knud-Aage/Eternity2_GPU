package dk.puzzle.blackwood;

import java.util.ArrayList;
import java.util.List;

public final class BwGpuTables {

    public static final int NUM_TABLES = 10;
    public static final int KEY_SPACE = 529;

    public static final int TABLE_CORNERS = 0;
    public static final int TABLE_LEFT_SIDES = 1;
    public static final int TABLE_TOP_SIDES = 2;
    public static final int TABLE_RIGHT_NOBREAK = 3;
    public static final int TABLE_RIGHT_BREAK = 4;
    public static final int TABLE_MIDDLES_NOBREAK = 5;
    public static final int TABLE_MIDDLES_BREAK = 6;
    public static final int TABLE_SOUTH_START = 7;
    public static final int TABLE_WEST_START = 8;
    public static final int TABLE_START = 9;

    public static final int TABLE_UNUSED_ROW0 = -1;

    private BwGpuTables() {
    }

    public record GpuTableSet(
            int[] csrOffset, int[] csrCount, int[] payload,
            int[] bottomRawOffset, int[] bottomRawCount, int[] bottomRawPayload,
            int[] stepToTableId, int[] stepBoardIdx,
            int[] breakArray, int[] heuristicArray) {
    }

    /**
     * Like {@link GpuTableSet}, but with {@code numVariants} independently-jittered copies of the
     * main payload instead of one -- see {@link #buildVariants}. csrOffset/csrCount are shared
     * across variants rather than duplicated, since bucket structure (which pieces geometrically
     * qualify, and how many) is determined by piece geometry, not by the random tie-breaking
     * jitter -- verified at build time, not just assumed, see buildVariants.
     */
    public record MultiVariantTableSet(
            int[] csrOffset, int[] csrCount, int[][] payloadVariants,
            int[] bottomRawOffset, int[] bottomRawCount, int[] bottomRawPayload,
            int[] stepToTableId, int[] stepBoardIdx,
            int[] breakArray, int[] heuristicArray) {
    }

    /** Packs one candidate into a single int32 -- see class BwGpuTables javadoc / plan for the bit layout. */
    static int pack(BwRotatedPiece p) {
        int b0 = p.pieceNumber() - 1;
        int b1 = p.topSide();
        int b2 = p.rightSide();
        int b3 = p.rotations() | (p.breakCount() << 2) | (p.heuristicSideCount() << 3);
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }

    /**
     * Inverse of {@link #pack}. Used to convert a GPU result board (produced by
     * {@code SolveBlackwoodKernel.cu}, same bit layout) back into {@link BwRotatedPiece}s, e.g.
     * for {@link BwUtil#saveBoard}. Empty cells (sentinel {@code -1}) are the caller's
     * responsibility to check for before calling this -- it doesn't special-case them.
     */
    public static BwRotatedPiece unpack(int rec) {
        int pieceNumber = (rec & 0xFF) + 1;
        int topSide = (rec >> 8) & 0xFF;
        int rightSide = (rec >> 16) & 0xFF;
        int rotations = (rec >> 24) & 0x3;
        int breakCount = (rec >> 26) & 0x1;
        int heuristicSideCount = (rec >> 27) & 0x7;
        return new BwRotatedPiece(pieceNumber, rotations, topSide, rightSide, breakCount, heuristicSideCount);
    }

    private static BwRotatedPiece[][][] tablesInOrder(BlackwoodSolver solver) {
        return new BwRotatedPiece[][][]{
                solver.corners, solver.leftSides, solver.topSides,
                solver.rightSidesWithoutBreaks, solver.rightSidesWithBreaks,
                solver.middlesNoBreak, solver.middlesWithBreak,
                solver.southStart, solver.westStart, solver.start
        };
    }

    public static GpuTableSet build(BlackwoodSolver solver) {
        BwRotatedPiece[][][] tables = tablesInOrder(solver);

        int[] csrOffset = new int[NUM_TABLES * KEY_SPACE];
        int[] csrCount = new int[NUM_TABLES * KEY_SPACE];
        List<Integer> payload = new ArrayList<>();
        for (int t = 0; t < NUM_TABLES; t++) {
            BwRotatedPiece[][] table = tables[t];
            for (int key = 0; key < KEY_SPACE; key++) {
                int idx = t * KEY_SPACE + key;
                BwRotatedPiece[] entries = table[key];
                if (entries == null || entries.length == 0) {
                    csrOffset[idx] = 0;
                    csrCount[idx] = 0;
                    continue;
                }
                csrOffset[idx] = payload.size();
                csrCount[idx] = entries.length;
                for (BwRotatedPiece e : entries) {
                    payload.add(pack(e));
                }
            }
        }

        // bottomSides is only ever queried at bottom=0 keys (row 0 requires south-facing==border
        // colour 0) -- and by data convention every rotation-0 edge-piece candidate already has
        // bottomSide==0 canonically (the same convention that makes topSides/leftSides/rightSides
        // work via a pure rotation filter with no separate border check -- see BlackwoodSolver
        // prepare()'s rotation filters), so bottomSidePiecesRotated's keys are ALL of this form
        // already; iterating left*23 here isn't a filter, it's exhaustive.
        int[] bottomRawOffset = new int[23];
        int[] bottomRawCount = new int[23];
        List<Integer> bottomPayload = new ArrayList<>();
        for (int left = 0; left < 23; left++) {
            int key = left * 23;
            List<BwUtil.RotatedCandidate> candidates = solver.bottomSidePiecesRotated.get(key);
            if (candidates == null || candidates.isEmpty()) {
                bottomRawOffset[left] = 0;
                bottomRawCount[left] = 0;
                continue;
            }
            bottomRawOffset[left] = bottomPayload.size();
            bottomRawCount[left] = candidates.size();
            for (BwUtil.RotatedCandidate c : candidates) {
                bottomPayload.add(pack(c.rotatedPiece()));
            }
        }

        // Reference-identity against masterPieceLookup, not a re-derived copy of the row/col
        // dispatch rules -- can never silently drift from what prepare() actually decided.
        int[] stepToTableId = new int[256];
        for (int step = 0; step < 256; step++) {
            int row = solver.boardOrderRow[step];
            int col = solver.boardOrderCol[step];
            if (row == 0) {
                stepToTableId[step] = TABLE_UNUSED_ROW0;
                continue;
            }
            BwRotatedPiece[][] table = solver.masterPieceLookup[row * 16 + col];
            int id = -1;
            for (int t = 0; t < NUM_TABLES; t++) {
                if (tables[t] == table) {
                    id = t;
                    break;
                }
            }
            if (id == -1) {
                throw new IllegalStateException("masterPieceLookup entry at row=" + row + " col=" + col
                        + " (step=" + step + ") doesn't reference-match any of the " + NUM_TABLES + " known tables");
            }
            stepToTableId[step] = id;
        }

        int[] stepBoardIdx = new int[256];
        for (int step = 0; step < 256; step++) {
            stepBoardIdx[step] = solver.boardOrderRow[step] * 16 + solver.boardOrderCol[step];
        }

        return new GpuTableSet(
                csrOffset, csrCount, toIntArray(payload),
                bottomRawOffset, bottomRawCount, toIntArray(bottomPayload),
                stepToTableId, stepBoardIdx,
                solver.breakArray.clone(), solver.heuristicArray.clone());
    }

    /**
     * Builds {@code numVariants} independently-jittered payload variants, each produced by the
     * exact same unmodified {@code solver.prepare()} + {@link #build} pipeline as the single-variant
     * path -- {@code prepare()} already draws a fresh, unseeded {@code Random} every call (see its
     * own comment), so calling it N times already gives N genuinely different, individually valid
     * candidate orderings. No new sorting or tie-breaking logic is introduced here; each variant is
     * exactly as correct (break-count-then-heuristic-count monotonic, per BlackwoodSolver's own
     * search-loop invariant) as the one table this project has always used, since it comes from the
     * identical code path.
     *
     * <p>This is what lets 16384 GPU threads stop sharing one frozen candidate ordering for an
     * entire epoch (up to 20,000 launches, roughly a day at current pace) -- each thread picks one
     * of these N variants for the lifetime of its attempt (see the kernel's persisted
     * {@code tableVariant}), instead of every thread making the identical greedy choice whenever it
     * reaches the identical board state in the ~94% of the board the main tables cover (the bottom
     * row already got independent per-thread jitter via bsPayload; this extends the same idea to
     * everything else).</p>
     *
     * <p>csrOffset/csrCount are taken from the first variant and asserted identical on every
     * subsequent one, since which pieces qualify for a bucket (and how many) is a structural,
     * geometry-only fact that jitter should never change -- checked here rather than assumed, so a
     * violated assumption fails loudly instead of silently corrupting the search.</p>
     */
    public static MultiVariantTableSet buildVariants(BlackwoodSolver solver, int numVariants) throws Exception {
        if (numVariants < 1) {
            throw new IllegalArgumentException("numVariants must be >= 1, got " + numVariants);
        }
        GpuTableSet first = build(solver);
        int[][] payloadVariants = new int[numVariants][];
        payloadVariants[0] = first.payload();
        for (int v = 1; v < numVariants; v++) {
            solver.prepare();
            GpuTableSet next = build(solver);
            if (next.payload().length != first.payload().length) {
                throw new IllegalStateException("payload length changed across prepare() calls: variant 0 had "
                        + first.payload().length + " entries, variant " + v + " had " + next.payload().length
                        + " -- structural bucket sizes should never depend on jitter");
            }
            if (!java.util.Arrays.equals(next.csrOffset(), first.csrOffset())
                    || !java.util.Arrays.equals(next.csrCount(), first.csrCount())) {
                throw new IllegalStateException("csrOffset/csrCount changed across prepare() calls at variant " + v
                        + " -- the assumption that bucket structure is jitter-independent does not hold");
            }
            payloadVariants[v] = next.payload();
        }
        return new MultiVariantTableSet(first.csrOffset(), first.csrCount(), payloadVariants,
                first.bottomRawOffset(), first.bottomRawCount(), first.bottomRawPayload(),
                first.stepToTableId(), first.stepBoardIdx(), first.breakArray(), first.heuristicArray());
    }

    private static int[] toIntArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
}
