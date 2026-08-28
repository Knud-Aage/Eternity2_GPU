package dk.puzzle.core;

import dk.puzzle.util.PieceUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

/**
 * Loads the 256-piece Eternity II set from {@code pieces.csv}.
 *
 * <p>Extracted verbatim from this project's original {@code Eternity} class, which was a
 * Swing GUI entry point. The solver only ever needed this one static method from it, so
 * carrying the whole GUI class (and everything it transitively pulled in) into this
 * repository would have meant shipping the entire evolutionary-solver and visualisation
 * stack to call a CSV reader.</p>
 *
 * <p>Format is TheSil's: one piece per line, {@code index,east,south,west,north}, with an
 * optional header line and empty colour fields meaning grey (border). Read by bare relative
 * path, so the process working directory must be the repository root.</p>
 */
public final class PieceLoader {

    private static final Logger logger = LogManager.getLogger(PieceLoader.class);

    private PieceLoader() {
    }

    public static int[] loadPieces() {
        try (BufferedReader br = new BufferedReader(new FileReader("pieces.csv"))) {
            int[] pieces = new int[256];
            int i = 0;
            String line;

            // Skip header line if present (starts with non-digit)
            br.mark(256);
            String firstLine = br.readLine();
            if (firstLine != null) {
                String trimmed = firstLine.trim();
                String[] headerPts = trimmed.split(",");
                boolean isHeader = headerPts.length <= 4 &&
                        headerPts[0].trim().equals("16");
                if (!isHeader) {
                    br.reset(); // not a header, rewind and process it
                }
            }

            while ((line = br.readLine()) != null && i < 256) {
                String trimmed = line.trim();
                if (trimmed.isEmpty()) continue;

                String[] pts = trimmed.split(",", -1); // -1 keeps trailing empty fields
                if (pts.length < 5) continue;

                int e = parseColorField(pts[1]);
                int s = parseColorField(pts[2]);
                int w = parseColorField(pts[3]);
                int n = parseColorField(pts[4]);

                pieces[i++] = PieceUtils.pack(n, e, s, w);
            }

            if (i == 256) {
                logger.info("Loaded {} pieces from pieces.csv.", i);
                return pieces;
            }
            logger.error("pieces.csv only contained {} pieces, expected 256 -- FALLING BACK TO MOCK DATA. "
                    + "Every result from this run is meaningless. Check the working directory.", i);
        } catch (Exception e) {
            logger.error("pieces.csv not found or unreadable ({}) -- FALLING BACK TO MOCK DATA. "
                    + "Every result from this run is meaningless. Run from the repository root.", e.getMessage());
        }
        return generateMock();
    }

    /**
     * Parses a color field from the CSV, returning 0 for empty or missing fields.
     * TheSil's format uses empty fields for grey (border) edges.
     */
    private static int parseColorField(String field) {
        if (field == null) return 0;
        String trimmed = field.trim();
        if (trimmed.isEmpty()) return 0;
        try {
            return Integer.parseInt(trimmed);
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /**
     * A random, self-consistent 16x16 piece set, used only when pieces.csv can't be read.
     *
     * <p>Kept for behavioural parity with the original, but the callers above now log the
     * fallback at ERROR rather than INFO: a silent downgrade to mock data has bitten this
     * project before (a tool invoked with the wrong working directory produced plausible
     * but meaningless conflict counts for hours before anyone noticed).</p>
     */
    private static int[] generateMock() {
        int dim = 16;
        int[][] hEdges = new int[dim][dim + 1];
        int[][] vEdges = new int[dim + 1][dim];
        Random rnd = new Random(99);
        int interiorColors = 4;
        for (int r = 0; r < dim; r++) for (int c = 1; c < dim; c++) hEdges[r][c] = rnd.nextInt(interiorColors) + 1;
        for (int r = 1; r < dim; r++) for (int c = 0; c < dim; c++) vEdges[r][c] = rnd.nextInt(interiorColors) + 1;
        int[] pieces = new int[256];
        int idx = 0;
        for (int r = 0; r < dim; r++) {
            for (int c = 0; c < dim; c++) {
                pieces[idx++] = PieceUtils.pack(vEdges[r][c], hEdges[r][c + 1], vEdges[r + 1][c], hEdges[r][c]);
            }
        }
        return pieces;
    }
}
