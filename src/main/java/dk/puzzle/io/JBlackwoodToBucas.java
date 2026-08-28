package dk.puzzle.io;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class JBlackwoodToBucas {

    public static void main(String[] args) {
        String piecesFile = "src\\main\\resources\\JBlackwood_Pieces.txt";
        String boardFile = "src\\main\\resources\\Raw_468_Board.txt";
        String outputFile = "records\\TYPEWRITER_LOCKED\\Record_468Pieces.csv";

        try {
            // 1. Load his exact colors from the C# definitions
            int[][] jblackwoodPieces = loadHisPieces(piecesFile);

            // 2. Read his board layout
            String rawBoard = new String(Files.readAllBytes(Paths.get(boardFile)));

            // 3. Convert and generate CSV
            generateCsv(rawBoard, jblackwoodPieces, outputFile);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Parses the C# Piece objects directly into a 2D Java int array.
     * Looks for: PieceNumber = X, TopSide = N, RightSide = E, BottomSide = S, LeftSide = W
     */
    public static int[][] loadHisPieces(String filepath) throws IOException {
        int[][] pieces = new int[256][4];
        List<String> lines = Files.readAllLines(Paths.get(filepath));

        // Regex pattern to extract the 5 numbers from his C# formatting
        Pattern p = Pattern.compile("PieceNumber\\s*=\\s*(\\d+).*?TopSide\\s*=\\s*(\\d+).*?RightSide\\s*=\\s*(\\d+).*?BottomSide\\s*=\\s*(\\d+).*?LeftSide\\s*=\\s*(\\d+)");

        int loaded = 0;
        for (String line : lines) {
            Matcher m = p.matcher(line);
            if (m.find()) {
                int id = Integer.parseInt(m.group(1)); // 1 to 256
                int top = Integer.parseInt(m.group(2));
                int right = Integer.parseInt(m.group(3));
                int bottom = Integer.parseInt(m.group(4));
                int left = Integer.parseInt(m.group(5));

                // Store in our 0-indexed array (ID 1 goes to index 0)
                int arrayIndex = id - 1;
                pieces[arrayIndex][0] = top;    // North
                pieces[arrayIndex][1] = right;  // East
                pieces[arrayIndex][2] = bottom; // South
                pieces[arrayIndex][3] = left;   // West

                loaded++;
            }
        }
        System.out.println("Loaded " + loaded + " pieces from JBlackwood definitions.");
        return pieces;
    }

    private static void generateCsv(String rawBoard, int[][] piecesDb, String outputPath) throws IOException {
        try (PrintWriter writer = new PrintWriter(new File(outputPath))) {
            writer.println("Row,Col,North,East,South,West");

            String[] lines = rawBoard.trim().split("\\r?\\n");

            for (int row = 0; row < 16; row++) {
                if (row >= lines.length) break;

                String[] tokens = lines[row].trim().split("\\s+");
                for (int col = 0; col < 16; col++) {
                    if (col >= tokens.length) break;

                    String token = tokens[col];
                    if (token.equals("---/-")) {
                        continue; // Skip the empty hole
                    }

                    // Parse "ID/Rotation"
                    String[] parts = token.split("/");
                    int physicalId = Integer.parseInt(parts[0]);
                    int rotation = Integer.parseInt(parts[1]);

                    // Get his base colors
                    int[] baseColors = piecesDb[physicalId - 1];
                    int n = baseColors[0], e = baseColors[1], s = baseColors[2], w = baseColors[3];

                    // Apply clockwise rotation
                    int rN = n, rE = e, rS = s, rW = w;
                    if (rotation == 1) { // 90 deg clockwise
                        rN = w; rE = n; rS = e; rW = s;
                    } else if (rotation == 2) { // 180 deg
                        rN = s; rE = w; rS = n; rW = e;
                    } else if (rotation == 3) { // 270 deg clockwise
                        rN = e; rE = s; rS = w; rW = n;
                    }

                    writer.printf("%d,%d,%d,%d,%d,%d%n", row, col, rN, rE, rS, rW);
                }
            }
            System.out.println("Successfully generated perfect Bucas CSV: " + outputPath);
        }
    }
}