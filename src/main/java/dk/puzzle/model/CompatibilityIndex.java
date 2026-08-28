package dk.puzzle.model;

import java.util.BitSet;

/**
 * Precomputed bitmask index for fast candidate lookup during DFS.
 *
 * Instead of iterating piecesByNorth[c], piecesByWest[c] etc. separately,
 * this class answers "which orientation indices satisfy ALL known constraints
 * at once" using BitSet.and() — a single operation replacing 4 array scans.
 *
 * Orientation indices match inventory.allOrientations[0..1023].
 *
 * Edge colour encoding:
 *   - 0        = border colour (used on outer edges)
 *   - 1..22    = inner colours
 *   - WILDCARD = -1 (don't care — neighbour not yet placed)
 */
public class CompatibilityIndex {

    // Number of distinct edge colours (0 = border, 1-22 = inner)
    private static final int NUM_COLORS = 23;
    // Total orientation slots (256 pieces × 4 rotations)
    private static final int TOTAL_ORIENTATIONS = 1024;

    public static final int WILDCARD = -1;
    public static final int N = 0, E = 1, S = 2, W = 3;

    // byEdge[side][colour] → BitSet of orientation indices that have 'colour' on 'side'
    private final BitSet[][] byEdge = new BitSet[4][NUM_COLORS];

    // physicalBit[orientationIdx] = the physical piece id for that orientation
    // Used to filter already-used pieces.
    private final int[] physicalMapping;

    // physicalMask[physId] = BitSet of all orientation indices belonging to that piece.
    // Lets us build a "used orientations" BitSet cheaply.
    private final BitSet[] physicalMask = new BitSet[256];

    public CompatibilityIndex(int[] allOrientations, int[] physicalMapping) {
        this.physicalMapping = physicalMapping;

        for (int side = 0; side < 4; side++)
            for (int c = 0; c < NUM_COLORS; c++)
                byEdge[side][c] = new BitSet(TOTAL_ORIENTATIONS);

        for (int i = 0; i < 256; i++)
            physicalMask[i] = new BitSet(TOTAL_ORIENTATIONS);

        for (int i = 0; i < TOTAL_ORIENTATIONS; i++) {
            int p = allOrientations[i];
            int physId = physicalMapping[i];

            int north = (p >>> 24) & 0xFF;
            int east  = (p >>> 16) & 0xFF;
            int south = (p >>>  8) & 0xFF;
            int west  =  p         & 0xFF;

            byEdge[N][north].set(i);
            byEdge[E][east ].set(i);
            byEdge[S][south].set(i);
            byEdge[W][west ].set(i);

            physicalMask[physId].set(i);
        }
    }

    /**
     * Returns a BitSet of all orientation indices compatible with the given
     * edge constraints. Pass WILDCARD (-1) for unknown sides.
     *
     * The caller must still filter out used physical pieces via
     * andNotUsed(result, localUsed).
     */
    public BitSet candidatesFor(int northColor, int eastColor, int southColor, int westColor) {
        // Start with the tightest constraint to minimise work for subsequent ands.
        // In practice all four sides matter — order doesn't change correctness.
        BitSet result = new BitSet(TOTAL_ORIENTATIONS);
        result.set(0, TOTAL_ORIENTATIONS);

        if (northColor != WILDCARD) result.and(byEdge[N][northColor]);
        if (eastColor  != WILDCARD) result.and(byEdge[E][eastColor]);
        if (southColor != WILDCARD) result.and(byEdge[S][southColor]);
        if (westColor  != WILDCARD) result.and(byEdge[W][westColor]);

        return result;
    }

    /**
     * Removes orientation indices whose physical piece is already used.
     * Mutates the passed-in BitSet in place for zero allocation in the hot loop.
     */
    public void andNotUsed(BitSet candidates, boolean[] localUsed) {
        for (int physId = 0; physId < 256; physId++) {
            if (localUsed[physId]) {
                candidates.andNot(physicalMask[physId]);
            }
        }
    }

    /**
     * Quick check: does ANY unused orientation satisfy the given constraints?
     * Used for forward lookahead without materialising the full candidate set.
     */
    public boolean hasAnyCandidate(int northColor, int eastColor, int southColor, int westColor,
                                   boolean[] localUsed) {
        BitSet result = candidatesFor(northColor, eastColor, southColor, westColor);
        andNotUsed(result, localUsed);
        return !result.isEmpty();
    }

}