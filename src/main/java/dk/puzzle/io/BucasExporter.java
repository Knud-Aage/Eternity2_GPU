package dk.puzzle.io;

import dk.puzzle.util.PieceUtils; // Sørg for at denne sti passer til din mappestruktur!
import java.io.BufferedReader;
import java.io.FileReader;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


/**
 * <p>A utility class designed to convert Eternity II puzzle solution records
 * from a CSV file format into a URL compatible with the <a href="https://e2.bucas.name/">Bucas Eternity II solver/viewer</a>.</p>
 *
 * <p>This exporter reads a CSV file, which can be in one of two formats:
 * a standard 16x16 grid format or a "macro" format (representing a 4x4 grid of 4x4 sub-grids).
 * It then processes each line to extract piece information (position and side patterns)
 * and constructs a 256-character string representing the entire 16x16 board.
 * Each piece's four sides (North, East, South, West) are encoded as lowercase
 * letters ('a' through 'w') based on their pattern ID.</p>
 *
 * <p>The generated URL allows users to visualize their puzzle solutions directly
 * in the Bucas online tool.</p>
 */
public class BucasExporter {

    private static final Logger logger = LogManager.getLogger(BucasExporter.class);

    /**
     * The main entry point for the BucasExporter application.
     *
     * <p>This method reads a specified CSV file containing Eternity II puzzle piece data,
     * processes it to determine piece positions and patterns, and then generates
     * a Bucas-compatible URL that can be used to view the puzzle solution online.</p>
     *
     * <p>It supports two input CSV formats: a standard 16x16 grid format and a "macro" format.
     * The format is detected by checking if the first line of the CSV contains "macro".</p>
     *
     * @param args Command line arguments (not used in this application).
     */
    /**
     * Remaps TheSil color numbers (0-22) to KnudHansen/Bucas color numbers.
     * Derived by comparing Thomas Egense's 22-piece solution with TheSil's data.
     */
    private static final int[] THESIL_TO_THOMAS = {
            0, 1, 3, 4, 2, 5, 6, 7, 9, 12, 14, 15, 19, 21, 8, 10, 13, 16, 17, 18, 20, 22, 11
    };

    public static String exportBoard(int[] board) {
        StringBuilder bucasString = new StringBuilder();

        for (int i = 0; i < 256; i++) {
            int p = board[i];

            if (p == -1 || p == -2) {
                bucasString.append("aaaa");
            } else {
                int n = PieceUtils.getNorth(p);
                int e = PieceUtils.getEast(p);
                int s = PieceUtils.getSouth(p);
                int w = PieceUtils.getWest(p);

                // Remap TheSil colors to KnudHansen colors for Bucas verification
                bucasString.append((char) ('a' + THESIL_TO_THOMAS[n]));
                bucasString.append((char) ('a' + THESIL_TO_THOMAS[e]));
                bucasString.append((char) ('a' + THESIL_TO_THOMAS[s]));
                bucasString.append((char) ('a' + THESIL_TO_THOMAS[w]));
            }
        }

        return "https://e2.bucas.name/#puzzle=KnudHansen&board_w=16&board_h=16&board_edges="
                + bucasString.toString();
    }

    // ==========================================================
    // DIN OPRINDELIGE METODE (Til CSV-filer)
    // ==========================================================
    public static void main(String[] args) {
//        String filename = "records/SPIRAL/Record_209Pieces.csv";
        String filename = "records/TYPEWRITER_LOCKED/Record_254Pieces.csv";

        String[] board = new String[256];
        for (int i = 0; i < 256; i++) {
            board[i] = "0000";
        }

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine();
            if (line == null) {
                return;
            }

            boolean isMacroFormat = line.toLowerCase().contains("macro");

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 6) {
                    int pos1 = Integer.parseInt(parts[0].trim());
                    int pos2 = Integer.parseInt(parts[1].trim());

                    int absoluteIndex = 0;
                    if (isMacroFormat) {
                        int row = (pos1 / 4) * 4 + (pos2 / 4);
                        int col = (pos1 % 4) * 4 + (pos2 % 4);
                        absoluteIndex = row * 16 + col;
                    } else {
                        absoluteIndex = pos1 * 16 + pos2;
                    }

                    int n = Integer.parseInt(parts[2].trim());

                    if (n >= 0 && n <= 22) {
                        int e = Integer.parseInt(parts[3].trim());
                        int s = Integer.parseInt(parts[4].trim());
                        int w = Integer.parseInt(parts[5].trim());

                        String pieceStr = "" +
                                (char) ('a' + n) +
                                (char) ('a' + e) +
                                (char) ('a' + s) +
                                (char) ('a' + w);

                        board[absoluteIndex] = pieceStr;
                    }
                }
            }

            StringBuilder bucasString = new StringBuilder();
            for (int i = 0; i < 256; i++) {
                bucasString.append(board[i]);
            }

            String finalUrl = "https://e2.bucas.name/#puzzle=Joshua_Blackwood_470&board_w=16&board_h=16&board_edges="
                    + bucasString
                    + "&motifs_order=jblackwood";

            logger.info("Finished. Press on this link to see the puzzle:");
            logger.info("------------------------------------------------------");
            logger.info(finalUrl);
            logger.info("------------------------------------------------------");

        } catch (Exception e) {
            logger.info("Couldn't find the file. Be sure that it is placed in the records folder.");
            e.printStackTrace();
        }
    }
}