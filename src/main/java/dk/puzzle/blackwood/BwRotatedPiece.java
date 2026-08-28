package dk.puzzle.blackwood;

/**
 * A piece already resolved to a specific rotation for a specific board cell.
 * Mirrors his C# {@code RotatedPiece} struct — deliberately omits left/bottom
 * side colours, which are recovered implicitly from already-placed
 * west/south-ish neighbours during search rather than stored per-cell.
 */
public record BwRotatedPiece(int pieceNumber, int rotations, int topSide, int rightSide,
                              int breakCount, int heuristicSideCount) {

    /** Empty-cell sentinel, mirrors C#'s zero-initialized stackalloc RotatedPiece. */
    public static final BwRotatedPiece EMPTY = new BwRotatedPiece(0, 0, 0, 0, 0, 0);
}
