package dk.puzzle.blackwood;

public record BwRotatedPiece(int pieceNumber, int rotations, int topSide, int rightSide,
                              int breakCount, int heuristicSideCount) {

    public static final BwRotatedPiece EMPTY = new BwRotatedPiece(0, 0, 0, 0, 0, 0);
}
