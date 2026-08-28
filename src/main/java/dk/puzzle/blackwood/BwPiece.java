package dk.puzzle.blackwood;

/**
 * A single physical piece in Joshua Blackwood's own raw colour numbering
 * (0-22) — NOT this project's TheSil numbering. Mirrors his C# {@code Piece}
 * struct field-for-field.
 */
public record BwPiece(int pieceNumber, int topSide, int rightSide, int bottomSide, int leftSide) {

    /** 2 = corner (#1-4), 1 = edge (#5-60), 0 = middle (#61-256). Mirrors Piece.PieceType(). */
    public int pieceType() {
        if (pieceNumber >= 1 && pieceNumber <= 4) {
            return 2;
        } else if (pieceNumber >= 5 && pieceNumber <= 60) {
            return 1;
        }
        return 0;
    }
}
