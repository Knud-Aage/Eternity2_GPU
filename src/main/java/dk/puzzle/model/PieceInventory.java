package dk.puzzle.model;

import dk.puzzle.util.PieceUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Manages the collection of Eternity II puzzle pieces.
 * Pre-calculates rotations, categorizes pieces by type (Corner, Edge, Interior),
 * and maintains compatibility tables for fast lookup.
 */
public class PieceInventory {
    public int[] allOrientations = new int[1024];
    public int[] physicalMapping = new int[1024];

    public final List<Integer> corners = new ArrayList<>();
    public final List<Integer> edges = new ArrayList<>();
    public final List<Integer> interior = new ArrayList<>();

    public final Map<Integer, List<Integer>> northEdgeToOrientations = new HashMap<>();
    public final Map<Integer, List<Integer>> eastEdgeToOrientations = new HashMap<>();
    public final Map<Integer, List<Integer>> southEdgeToOrientations = new HashMap<>();
    public final Map<Integer, List<Integer>> westEdgeToOrientations = new HashMap<>();


    public List<Integer>[][] compatibility;
    public int[] colorFrequency;

    /**
     * Initializes the inventory from base physical pieces.
     *
     * @param basePieces Array of 256 physical pieces (North, East, South, West).
     */
    @SuppressWarnings("unchecked")
    public PieceInventory(int[] basePieces) {
        int maxColor = 0;
        for (int p : basePieces) {
            maxColor = Math.max(maxColor, Math.max(PieceUtils.getNorth(p), Math.max(PieceUtils.getEast(p),
                    Math.max(PieceUtils.getSouth(p), PieceUtils.getWest(p)))));
        }

        int colorCount = maxColor + 1;
        compatibility = new ArrayList[4][colorCount];
        colorFrequency = new int[colorCount];
        for (int i = 0; i < 4; i++) {
            for (int c = 0; c < colorCount; c++) compatibility[i][c] = new ArrayList<>();
        }

        for (int i = 0; i < 256; i++) {
            int oriented = basePieces[i];
            for (int r = 0; r < 4; r++) {
                int orientationIdx = i * 4 + r;
                allOrientations[orientationIdx] = oriented;
                physicalMapping[orientationIdx] = i;

                // Populate compatibility (if still used elsewhere)
                compatibility[0][PieceUtils.getNorth(oriented)].add(orientationIdx);
                compatibility[1][PieceUtils.getEast(oriented)].add(orientationIdx);
                compatibility[2][PieceUtils.getSouth(oriented)].add(orientationIdx);
                compatibility[3][PieceUtils.getWest(oriented)].add(orientationIdx);

                // Populate new edge-to-orientations maps
                northEdgeToOrientations.computeIfAbsent(PieceUtils.getNorth(oriented), k -> new ArrayList<>()).add(orientationIdx);
                eastEdgeToOrientations.computeIfAbsent(PieceUtils.getEast(oriented), k -> new ArrayList<>()).add(orientationIdx);
                southEdgeToOrientations.computeIfAbsent(PieceUtils.getSouth(oriented), k -> new ArrayList<>()).add(orientationIdx);
                westEdgeToOrientations.computeIfAbsent(PieceUtils.getWest(oriented), k -> new ArrayList<>()).add(orientationIdx);


                colorFrequency[PieceUtils.getNorth(oriented)]++;
                colorFrequency[PieceUtils.getEast(oriented)]++;
                colorFrequency[PieceUtils.getSouth(oriented)]++;
                colorFrequency[PieceUtils.getWest(oriented)]++;

                int borders = PieceUtils.getBorderCount(oriented);
                if (borders == 2) {
                    corners.add(orientationIdx);
                } else if (borders == 1) {
                    edges.add(orientationIdx);
                } else {
                    interior.add(orientationIdx);
                }

                oriented = PieceUtils.rotate(oriented);
            }
        }
    }

    /**
     * Returns the base colors (North, East, South, West) for a given physical piece ID.
     * * @param physicalId The 1-based ID from the imported data (1 to 256).
     * @return An array of 4 integers representing {North, East, South, West}.
     */
    public int[] getBaseColors(int physicalId) {
        if (physicalId < 1 || physicalId > 256) {
            throw new IllegalArgumentException(
                    "physicalId must be a 1-based ID in [1, 256], got: " + physicalId);
        }

        // 1. Convert the 1-based ID from the text file to your 0-based Java index
        int arrayIndex = physicalId - 1;

        // 2. Find the base orientation (rotation 0) in your 1024-array.
        // Assuming rotation 0 is stored at (index * 4).
        int bitPackedPiece = this.allOrientations[arrayIndex * 4];

        // 3. Unpack the colors using your existing PieceUtils
        int n = PieceUtils.getNorth(bitPackedPiece);
        int e = PieceUtils.getEast(bitPackedPiece);
        int s = PieceUtils.getSouth(bitPackedPiece);
        int w = PieceUtils.getWest(bitPackedPiece);

        return new int[]{n, e, s, w};
    }
}
