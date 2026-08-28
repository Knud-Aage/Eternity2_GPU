package dk.puzzle.blackwood;

import java.util.ArrayList;
import java.util.List;

/**
 * Flattens an already-{@link BlackwoodSolver#prepare()}d solver's candidate tables into the
 * CSR (offset+count index, in {@code __constant__} memory) plus flat payload (in global device
 * memory) layout {@code SolveBlackwoodKernel.cu} expects. Pure data transformation, no CUDA calls
 * -- fully unit-testable without a GPU.
 *
 * <p>Deliberately reuses {@link BlackwoodSolver#prepare()}'s already-verified table fields
 * directly rather than re-deriving table construction a second time -- the single biggest
 * fidelity-risk reducer for the GPU port, since the kernel then sees exactly the same tables
 * the verified CPU port trusts.</p>
 */
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

    /** row==0 steps bypass c_stepToTableId entirely (handled directly in the kernel, mirroring
     *  BlackwoodSolver.solvePuzzle()'s own row==0 special case) -- this value is never read. */
    public static final int TABLE_UNUSED_ROW0 = -1;

    private BwGpuTables() {
    }

    public record GpuTableSet(
            int[] csrOffset, int[] csrCount, int[] payload,
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

    /** The 10 batch-level tables, in the fixed order the TABLE_* constants index into. */
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

    private static int[] toIntArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
}
