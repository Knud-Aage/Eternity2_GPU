package dk.puzzle.util;

/**
 * Utility class for manipulating Eternity II puzzle pieces using bitwise operations.
 * Pieces are represented as 32-bit packed integers where each 8-bit segment represents
 * a color ID for one of the four edges.
 * <p>
 * Bit layout: [North (31-24) | East (23-16) | South (15-8) | West (7-0)]
 */
public class PieceUtils {
    public static final int BORDER_COLOR = 0;
    public static final int WILDCARD = 0xFF;

    /**
     * Packs four 8-bit color IDs into a single 32-bit integer representation.
     *
     * @param n the North color ID (0-255)
     * @param e the East color ID (0-255)
     * @param s the South color ID (0-255)
     * @param w the West color ID (0-255)
     * @return a 32-bit packed integer representation of the piece
     */
    public static int pack(int n, int e, int s, int w) {
        return ((n & 0xFF) << 24) | ((e & 0xFF) << 16) | ((s & 0xFF) << 8) | (w & 0xFF);
    }

    /**
     * Extracts the North color ID from a packed piece.
     *
     * @param p the 32-bit packed integer representation of a piece
     * @return the 8-bit North color ID
     */
    public static int getNorth(int p) {
        return (p >>> 24) & 0xFF;
    }

    /**
     * Extracts the East color ID from a packed piece.
     *
     * @param p the 32-bit packed integer representation of a piece
     * @return the 8-bit East color ID
     */
    public static int getEast(int p) {
        return (p >>> 16) & 0xFF;
    }

    /**
     * Extracts the South color ID from a packed piece.
     *
     * @param p the 32-bit packed integer representation of a piece
     * @return the 8-bit South color ID
     */
    public static int getSouth(int p) {
        return (p >>> 8) & 0xFF;
    }

    /**
     * Extracts the West color ID from a packed piece.
     *
     * @param p the 32-bit packed integer representation of a piece
     * @return the 8-bit West color ID
     */
    public static int getWest(int p) {
        return p & 0xFF;
    }

    /**
     * Rotates the given packed piece 90 degrees clockwise.
     * The mapping follows: New North = Old West, New East = Old North,
     * New South = Old East, New West = Old South.
     *
     * @param p the 32-bit packed integer to rotate
     * @return the rotated 32-bit packed integer
     */
    public static int rotate(int p) {
        int n = (p >>> 24) & 0xFF;
        int e = (p >>> 16) & 0xFF;
        int s = (p >>> 8) & 0xFF;
        int w = p & 0xFF;
        return pack(w, n, e, s);
    }

    /**
     * Counts how many edges of a piece match the {@link #BORDER_COLOR}.
     *
     * @param p the 32-bit packed integer representation of a piece
     * @return the count of border edges (0 to 4)
     */
    public static int getBorderCount(int p) {
        int count = 0;
        if (getNorth(p) == BORDER_COLOR) {
            count++;
        }
        if (getEast(p) == BORDER_COLOR) {
            count++;
        }
        if (getSouth(p) == BORDER_COLOR) {
            count++;
        }
        if (getWest(p) == BORDER_COLOR) {
            count++;
        }
        return count;
    }
}
