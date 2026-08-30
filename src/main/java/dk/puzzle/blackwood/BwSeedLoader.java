package dk.puzzle.blackwood;

import dk.puzzle.model.PieceInventory;
import dk.puzzle.tools.HoleSolver;
import dk.puzzle.util.PieceUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Loads previously saved deep boards and converts them into the step-ordered form the GPU kernel
 * replays from, so a run can start at the frontier instead of re-deriving 250 pieces of
 * already-known progress from scratch every time.
 *
 * <p>Reads the grid written by {@link BwUtil#buildBoardString} -- 16 lines, <b>board row 15
 * first</b>, cells of the form {@code "NNN/R"} (piece number / rotation) or {@code "---/-"} for
 * empty. The C# solver's own {@code Save_Board} emits a byte-identical grid layout, so boards from
 * either solver load through this same parser; only the trailing bucas link differs, and that is
 * ignored here.</p>
 *
 * <p>The kernel needs pieces in SEARCH-STEP order, not board order, because it replays step 0, 1,
 * 2... along Blackwood's inward spiral. Board index for each step comes from
 * {@code GpuTableSet.stepBoardIdx}. A seed is truncated at the first step whose board cell is
 * empty: the kernel can resume from a prefix, but a hole mid-sequence would leave later pieces
 * with no valid neighbour context.</p>
 */
public final class BwSeedLoader {

    private static final Logger logger = LogManager.getLogger(BwSeedLoader.class);

    private BwSeedLoader() {
    }

    /**
     * One saved board, already converted to the kernel's step-ordered encoding.
     *
     * @param link      the board's own bucas link, straight from the file's last line -- kept so
     *                  {@link #rankByConflicts} can score the board without reconstructing it
     * @param conflicts what this board completes to via HoleSolver, or -1 if not scored yet.
     *                  Depth alone is a poor quality signal (a 252-piece board has been measured
     *                  completing to 13 conflicts while a 251-piece one completes to 12), so an
     *                  unscored pool would resume from mediocre boards as readily as good ones.
     */
    public record Seed(Path source, int depth, int[] stepEncoded, String link, int conflicts) {
    }

    private static final java.util.regex.Pattern LEGACY_NAME =
            java.util.regex.Pattern.compile("^\\d+_[0-9a-fA-F-]+_\\d+\\.txt$");
    private static final java.util.regex.Pattern BASEBOARD_NAME =
            java.util.regex.Pattern.compile("^Errors\\d+_Base(\\d+)_.*_baseboard\\.txt$");

    /**
     * Deterministic selection: the {@code maxSeeds} deepest boards. Kept for the measurement
     * harnesses, which need a fixed pool to compare runs against.
     */
    public static List<Seed> load(List<Path> dirs, int minDepth, int maxSeeds, int[] stepBoardIdx) {
        return load(dirs, minDepth, maxSeeds, stepBoardIdx, null);
    }

    /**
     * As {@link #load(List, int, int, int[])}, but when {@code rand} is non-null the pool is drawn
     * by depth-weighted random sampling instead of taking a strict top-{@code maxSeeds}.
     *
     * <p>Strict top-K makes the pool a pure function of what is on disk, so every process restart
     * resumed from the identical elite boards and re-mined neighbourhoods already exhausted. That
     * showed up directly in the results: of 18 saved 12-conflict boards, only 9 were distinct, and
     * every duplicate spanned a restart. Sampling gives each run a different slice of the candidate
     * pool while still strongly favouring deep boards.</p>
     */
    public static List<Seed> load(List<Path> dirs, int minDepth, int maxSeeds, int[] stepBoardIdx,
                                  Random rand) {
        List<Seed> seeds = new ArrayList<>();
        for (Path dir : dirs) {
            if (!Files.isDirectory(dir)) {
                logger.debug("Seed directory {} does not exist, skipping", dir);
                continue;
            }
            int loadedHere = 0;
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.txt")) {
                for (Path file : stream) {
                    // Cheap pre-filter on filename-encoded depth, so a directory of thousands of
                    // shallow or wrong-format boards isn't fully parsed just to reject it.
                    if (depthFromFilename(file) < minDepth) continue;
                    Seed seed = parse(file, stepBoardIdx);
                    if (seed != null && seed.depth() >= minDepth) {
                        seeds.add(seed);
                        loadedHere++;
                    }
                }
            } catch (IOException e) {
                logger.warn("Could not scan seed directory {}", dir, e);
            }
            if (loadedHere > 0) logger.info("Loaded {} candidate seed board(s) from {}", loadedHere, dir);
        }

        if (seeds.size() <= maxSeeds) {
            seeds.sort(Comparator.comparingInt(Seed::depth).reversed());
            return seeds;
        }
        if (rand == null) {
            seeds.sort(Comparator.comparingInt(Seed::depth).reversed());
            return new ArrayList<>(seeds.subList(0, maxSeeds));
        }
        return sampleWeightedByDepth(seeds, minDepth, maxSeeds, rand);
    }

    /**
     * Exponent applied to a board's depth advantage when weighting it for selection. At the default
     * of 3 and a 245 floor, a 253-piece board is {@code 9^3 = 729} times likelier to be drawn than a
     * 245-piece one -- deep boards still dominate the pool, but the pool is no longer the same set
     * on every restart.
     */
    private static final double DEPTH_BIAS_EXPONENT = 3.0;

    /**
     * Weighted sampling without replacement, Efraimidis-Spirakis: draw {@code key = ln(U) / weight}
     * per item and keep the largest {@code maxSeeds} keys. Using the log form rather than
     * {@code U^(1/weight)} keeps it numerically stable at the large weights the exponent produces.
     */
    static List<Seed> sampleWeightedByDepth(List<Seed> seeds, int minDepth, int maxSeeds,
                                            Random rand) {
        record Keyed(double key, Seed seed) {
        }
        List<Keyed> keyed = new ArrayList<>(seeds.size());
        for (Seed seed : seeds) {
            double weight = Math.pow(seed.depth() - minDepth + 1, DEPTH_BIAS_EXPONENT);
            // nextDouble() can return exactly 0.0; ln(0) would be -inf and permanently sink the item.
            double u = rand.nextDouble();
            if (u <= 0.0) u = Double.MIN_VALUE;
            keyed.add(new Keyed(Math.log(u) / weight, seed));
        }
        keyed.sort(Comparator.comparingDouble(Keyed::key).reversed());

        List<Seed> picked = new ArrayList<>(maxSeeds);
        for (int i = 0; i < maxSeeds; i++) {
            picked.add(keyed.get(i).seed());
        }
        picked.sort(Comparator.comparingInt(Seed::depth).reversed());
        return picked;
    }

    /** -1 for anything not in one of the two Blackwood-numbered conventions -- see LEGACY_NAME/BASEBOARD_NAME. */
    private static int depthFromFilename(Path file) {
        String name = file.getFileName().toString();

        java.util.regex.Matcher baseboard = BASEBOARD_NAME.matcher(name);
        if (baseboard.matches()) {
            return Integer.parseInt(baseboard.group(1));
        }

        if (LEGACY_NAME.matcher(name).matches()) {
            int underscore = name.indexOf('_');
            try {
                return Integer.parseInt(name.substring(0, underscore));
            } catch (NumberFormatException e) {
                return -1;
            }
        }

        return -1; // includes *_RawBoard.txt / *_physical_layout.txt -- wrong numbering, never a seed source
    }

    /** Returns null if the file is not a readable board grid. */
    static Seed parse(Path file, int[] stepBoardIdx) {
        List<String> lines;
        try {
            lines = Files.readAllLines(file);
        } catch (IOException e) {
            logger.warn("Could not read seed board {}", file, e);
            return null;
        }
        if (lines.size() < 16) return null;

        // board[boardIdx] = (pieceNumber << 2) | rotation, or -1 where empty.
        int[] byBoardIdx = new int[256];
        java.util.Arrays.fill(byBoardIdx, -1);

        for (int line = 0; line < 16; line++) {
            String[] cells = lines.get(line).trim().split("\\s+");
            if (cells.length < 16) return null;
            int boardRow = 15 - line; // the grid is written with row 15 first
            for (int col = 0; col < 16; col++) {
                String cell = cells[col];
                int slash = cell.indexOf('/');
                if (slash <= 0) return null;
                String pieceText = cell.substring(0, slash);
                String rotText = cell.substring(slash + 1);
                if (pieceText.startsWith("-") || rotText.startsWith("-")) continue; // empty cell
                try {
                    int pieceNumber = Integer.parseInt(pieceText.trim());
                    int rotation = Integer.parseInt(rotText.trim());
                    if (pieceNumber < 1 || pieceNumber > 256 || rotation < 0 || rotation > 3) return null;
                    byBoardIdx[boardRow * 16 + col] = (pieceNumber << 2) | rotation;
                } catch (NumberFormatException e) {
                    return null;
                }
            }
        }

        int[] stepEncoded = new int[256];
        java.util.Arrays.fill(stepEncoded, -1);
        int depth = 0;
        for (int step = 0; step < 256; step++) {
            int v = byBoardIdx[stepBoardIdx[step]];
            if (v < 0) break; // truncate at the first gap -- a prefix is resumable, a hole is not
            stepEncoded[step] = v;
            depth++;
        }
        if (depth == 0) return null;

        String link = null;
        for (int i = lines.size() - 1; i >= 16; i--) {
            int idx = lines.get(i).indexOf("https://");
            if (idx >= 0) { link = lines.get(i).substring(idx).trim(); break; }
        }
        return new Seed(file, depth, stepEncoded, link, -1);
    }

    /**
     * Scores each seed by what it actually completes to and returns the best {@code maxSeeds},
     * lowest conflicts first. This is what stops the pool from treating a 247-piece board that
     * completes to 21 conflicts as equal to a 250-piece board that completes to 13.
     *
     * <p>Runs HoleSolver's real completion once per candidate, which costs roughly a second each --
     * affordable because the pool is only rebuilt at an epoch boundary, and epochs are now long.
     * Seeds whose link is missing or unreadable keep conflicts = -1 and sort last rather than being
     * dropped, so a parse quirk degrades the ranking instead of silently shrinking the pool.</p>
     */
    public static List<Seed> rankByConflicts(List<Seed> seeds, PieceInventory inventory,
                                             int trials, int maxSeeds) {
        List<Seed> scored = new ArrayList<>(seeds.size());
        for (Seed seed : seeds) {
            int conflicts = Integer.MAX_VALUE;
            if (seed.link() != null) {
                try {
                    int[] decoded = HoleSolver.decodeBoardAuto(seed.link(), inventory, false);
                    HoleSolver.ConflictSolveResult result =
                            HoleSolver.solveConflicts(decoded, inventory, false, trials);
                    conflicts = countConflicts(result.bestBoard());
                } catch (Exception e) {
                    logger.debug("Could not score seed {}", seed.source().getFileName(), e);
                }
            }
            scored.add(new Seed(seed.source(), seed.depth(), seed.stepEncoded(), seed.link(), conflicts));
        }

        // Fewest conflicts wins; deeper breaks ties, since a deeper board of equal completed
        // quality leaves the GPU less ground to re-cover.
        scored.sort(Comparator.comparingInt(Seed::conflicts).thenComparing(Comparator.comparingInt(Seed::depth).reversed()));
        return scored.size() > maxSeeds ? new ArrayList<>(scored.subList(0, maxSeeds)) : scored;
    }

    private static int countConflicts(int[] board) {
        int conflicts = 0;
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                int i = r * 16 + c;
                if (c < 15 && PieceUtils.getEast(board[i]) != PieceUtils.getWest(board[i + 1])) conflicts++;
                if (r < 15 && PieceUtils.getSouth(board[i]) != PieceUtils.getNorth(board[i + 16])) conflicts++;
            }
        }
        return conflicts;
    }
}
