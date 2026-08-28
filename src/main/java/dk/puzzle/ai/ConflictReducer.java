package dk.puzzle.ai;

import dk.puzzle.model.PieceInventory;
import dk.puzzle.util.PieceUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.*;

/**
 * ConflictReducer: local search to minimise internal edge conflicts on a
 * near-complete board without adding or removing pieces.
 *
 * Three strategies:
 *
 * 1. ROTATION REPAIR — for each conflicted piece, try all 3 alternative
 *    rotations and keep whichever reduces total conflicts most. O(conflict_count * 4).
 *
 * 2. SWAP REPAIR — for each conflicted piece, try swapping it with every
 *    other placed piece and keep the swap if it strictly reduces conflicts.
 *    O(conflict_count * placed_count).
 *
 * Both strategies are conflict-neutral by default (only apply changes that
 * strictly reduce total conflict count), so the piece count never decreases.
 *
 * 3. RANDOM-RESTART FILL — for boards with remaining holes, repeatedly fills
 * every hole in a random order, each time placing whichever unused piece/
 * orientation best matches its neighbours, and keeps the best of many trials.
 * This is the automated version of manually re-filling a near-complete board
 * from scratch many times and keeping the luckiest attempt.
 */
public class ConflictReducer {

    private static final Logger logger = LogManager.getLogger(ConflictReducer.class);

    private final PieceInventory inventory;
    private final boolean lockCenter;

    public ConflictReducer(PieceInventory inventory, boolean lockCenter) {
        this.inventory = inventory;
        this.lockCenter = lockCenter;
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Run one pass of conflict reduction and return the improved board.
     * Suitable for calling after every Phase 3 cycle during solving.
     *
     * @param board      The board to improve (modified in-place AND returned).
     * @param maxPasses  How many rotation+swap passes to run (1-3 for live use).
     * @return           Total conflicts remaining after improvement.
     */
    public int reduceLive(int[] board, int maxPasses) {
        return reduce(board, maxPasses, false);
    }

    /**
     * Run an extended conflict reduction for post-processing a saved board.
     * Runs until no further improvement is found or maxPasses is reached.
     *
     * @param board      The board to improve (modified in-place AND returned).
     * @param maxPasses  Maximum passes (20-100 for post-processing).
     * @return           Total conflicts remaining.
     */
    public int reducePostProcess(int[] board, int maxPasses) {
        return reduce(board, maxPasses, true);
    }

    /**
     * Randomised-restart hole filler, for boards that still have empty positions
     * (marked -1 or -2). Each trial shuffles the holes into a random order and
     * fills them one at a time with whichever unused piece/orientation best
     * matches the neighbours already on the board (fewest edge mismatches),
     * ties broken randomly. The best board found across all trials (fewest
     * total conflicts) is then polished with {@link #reducePostProcess} and
     * returned.
     *
     * @param board  The board to fill (NOT modified in-place — a fully-filled
     *               copy is returned; pass the result to your record-saving code).
     * @param trials Number of random fill attempts (100,000+ recommended;
     *               each trial is cheap so millions are feasible for post-processing).
     * @return       The best fully-filled board found, already polished.
     */
    public int[] randomRestartFill(int[] board, int trials) {
        List<Integer> holePositions = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1 || board[i] == -2) holePositions.add(i);
        }
        if (holePositions.isEmpty()) return Arrays.copyOf(board, 256);

        boolean[] usedPhysical = new boolean[256];
        for (int i = 0; i < 256; i++) {
            int p = board[i];
            if (p != -1 && p != -2) {
                int physId = getPhysId(p);
                if (physId >= 0) usedPhysical[physId] = true;
            }
        }
        List<Integer> unusedPhysIds = new ArrayList<>();
        for (int physId = 0; physId < 256; physId++) {
            if (!usedPhysical[physId]) unusedPhysIds.add(physId);
        }

        Random rnd = new Random();
        int[] bestBoard = fillOnce(board, holePositions, unusedPhysIds, rnd);
        int bestConflicts = countConflicts(bestBoard);

        for (int t = 1; t < trials; t++) {
            int[] trial = fillOnce(board, holePositions, unusedPhysIds, rnd);
            int conflicts = countConflicts(trial);
            if (conflicts < bestConflicts) {
                bestConflicts = conflicts;
                bestBoard = trial;
                if (bestConflicts == 0) break;
            }
        }

        int polished = reducePostProcess(bestBoard, 50);
        logger.info(String.format(
                ">>> [RANDOM RESTART FILL] %d trials on %d holes → best fill %d conflicts → polished to %d.",
                trials, holePositions.size(), bestConflicts, polished));
        return bestBoard;
    }

    /**
     * Fills every hole once, in a random order, choosing at each hole the
     * unused piece/orientation with the fewest edge mismatches against
     * whatever neighbours are already placed. Ties are broken randomly.
     */
    private int[] fillOnce(int[] sourceBoard, List<Integer> holePositions, List<Integer> unusedPhysIds, Random rnd) {
        int[] trial = Arrays.copyOf(sourceBoard, 256);
        List<Integer> holes = new ArrayList<>(holePositions);
        Collections.shuffle(holes, rnd);
        List<Integer> remaining = new ArrayList<>(unusedPhysIds);

        for (int pos : holes) {
            int row = pos / 16, col = pos % 16;
            int northReq = (row == 0)  ? 0 : (isPlaced(trial[pos - 16]) ? PieceUtils.getSouth(trial[pos - 16]) : -1);
            int southReq = (row == 15) ? 0 : (isPlaced(trial[pos + 16]) ? PieceUtils.getNorth(trial[pos + 16]) : -1);
            int westReq  = (col == 0)  ? 0 : (isPlaced(trial[pos - 1])  ? PieceUtils.getEast(trial[pos - 1])   : -1);
            int eastReq  = (col == 15) ? 0 : (isPlaced(trial[pos + 1])  ? PieceUtils.getWest(trial[pos + 1])   : -1);

            int bestViolations = Integer.MAX_VALUE;
            List<int[]> bestCandidates = new ArrayList<>(); // {remainingIndex, orientedPiece}

            for (int k = 0; k < remaining.size(); k++) {
                int physId = remaining.get(k);
                for (int r = 0; r < 4; r++) {
                    int candidate = inventory.allOrientations[physId * 4 + r];

                    int violations = 0;
                    if (northReq != -1 && PieceUtils.getNorth(candidate) != northReq) violations++;
                    if (southReq != -1 && PieceUtils.getSouth(candidate) != southReq) violations++;
                    if (westReq  != -1 && PieceUtils.getWest(candidate)  != westReq)  violations++;
                    if (eastReq  != -1 && PieceUtils.getEast(candidate)  != eastReq)  violations++;

                    if (violations < bestViolations) {
                        bestViolations = violations;
                        bestCandidates.clear();
                        bestCandidates.add(new int[]{k, candidate});
                    } else if (violations == bestViolations) {
                        bestCandidates.add(new int[]{k, candidate});
                    }
                }
            }

            int[] chosen = bestCandidates.get(rnd.nextInt(bestCandidates.size()));
            trial[pos] = chosen[1];
            remaining.remove(chosen[0]);
        }

        return trial;
    }

    /**
     * Most-Constrained-Variable (MCV) hole filler: like {@link #randomRestartFill},
     * but instead of visiting holes in a random (or any fixed) order, it always
     * fills whichever remaining hole currently has the MOST already-determined
     * neighbour edges next, re-evaluating after every placement.
     *
     * <p>Why this matters: a fixed visiting order (row-by-row, or the back-to-
     * front approach used manually) leaves most cells in the interior of a hole
     * with only 1-2 of their 4 edges known when they're filled, because their
     * other neighbours are themselves still holes. MCV instead fills the edges
     * of the hole first (where 3 sides are already known from the solved
     * exterior), which pins down the interior cells' constraints one at a time
     * as it works inward — so by the time an interior cell is reached, far more
     * of its edges are already determined, and the greedy piece choice there is
     * far more likely to be conflict-free. Confirmed empirically: a colleague's
     * back-to-front fixed-order fill of a 16-cell hole bottomed out at 21
     * conflicts; the same hole analysed with MCV ordering is expected to do
     * better because it avoids committing to the least-constrained cells first.</p>
     *
     * @param board  The board to fill (NOT modified in-place).
     * @param trials Number of MCV fill attempts (ties within each fill are
     *               broken randomly, so repeated trials still explore
     *               different boards).
     * @return       The best fully-filled board found, already polished.
     */
    public int[] mcvRestartFill(int[] board, int trials) {
        List<Integer> holePositions = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            if (board[i] == -1 || board[i] == -2) holePositions.add(i);
        }
        if (holePositions.isEmpty()) return Arrays.copyOf(board, 256);

        boolean[] usedPhysical = new boolean[256];
        for (int i = 0; i < 256; i++) {
            int p = board[i];
            if (p != -1 && p != -2) {
                int physId = getPhysId(p);
                if (physId >= 0) usedPhysical[physId] = true;
            }
        }
        List<Integer> unusedPhysIds = new ArrayList<>();
        for (int physId = 0; physId < 256; physId++) {
            if (!usedPhysical[physId]) unusedPhysIds.add(physId);
        }

        Random rnd = new Random();
        int[] bestBoard = mcvFillOnce(board, holePositions, unusedPhysIds, rnd);
        int bestConflicts = countConflicts(bestBoard);

        for (int t = 1; t < trials; t++) {
            int[] trial = mcvFillOnce(board, holePositions, unusedPhysIds, rnd);
            int conflicts = countConflicts(trial);
            if (conflicts < bestConflicts) {
                bestConflicts = conflicts;
                bestBoard = trial;
                if (bestConflicts == 0) break;
            }
        }

        int polished = reducePostProcess(bestBoard, 50);
//        logger.info(String.format(
//                ">>> [MCV RESTART FILL] %d trials on %d holes → best fill %d conflicts → polished to %d.",
//                trials, holePositions.size(), bestConflicts, polished));
        return bestBoard;
    }

    /**
     * Fills every hole once, always choosing next whichever remaining hole has
     * the most already-determined neighbour edges (ties broken randomly), then
     * placing the unused piece/orientation with the fewest edge mismatches
     * against those known neighbours (ties broken randomly too).
     */
    private int[] mcvFillOnce(int[] sourceBoard, List<Integer> holePositions, List<Integer> unusedPhysIds, Random rnd) {
        int[] trial = Arrays.copyOf(sourceBoard, 256);
        Set<Integer> remainingHoles = new HashSet<>(holePositions);
        List<Integer> remaining = new ArrayList<>(unusedPhysIds);

        while (!remainingHoles.isEmpty()) {
            int pos = pickMostConstrainedHole(trial, remainingHoles, rnd);
            remainingHoles.remove(pos);

            int row = pos / 16, col = pos % 16;
            int northReq = (row == 0)  ? 0 : (isPlaced(trial[pos - 16]) ? PieceUtils.getSouth(trial[pos - 16]) : -1);
            int southReq = (row == 15) ? 0 : (isPlaced(trial[pos + 16]) ? PieceUtils.getNorth(trial[pos + 16]) : -1);
            int westReq  = (col == 0)  ? 0 : (isPlaced(trial[pos - 1])  ? PieceUtils.getEast(trial[pos - 1])   : -1);
            int eastReq  = (col == 15) ? 0 : (isPlaced(trial[pos + 1])  ? PieceUtils.getWest(trial[pos + 1])   : -1);

            int bestViolations = Integer.MAX_VALUE;
            List<int[]> bestCandidates = new ArrayList<>(); // {remainingIndex, orientedPiece}

            for (int k = 0; k < remaining.size(); k++) {
                int physId = remaining.get(k);
                for (int r = 0; r < 4; r++) {
                    int candidate = inventory.allOrientations[physId * 4 + r];

                    int violations = 0;
                    if (northReq != -1 && PieceUtils.getNorth(candidate) != northReq) violations++;
                    if (southReq != -1 && PieceUtils.getSouth(candidate) != southReq) violations++;
                    if (westReq  != -1 && PieceUtils.getWest(candidate)  != westReq)  violations++;
                    if (eastReq  != -1 && PieceUtils.getEast(candidate)  != eastReq)  violations++;

                    if (violations < bestViolations) {
                        bestViolations = violations;
                        bestCandidates.clear();
                        bestCandidates.add(new int[]{k, candidate});
                    } else if (violations == bestViolations) {
                        bestCandidates.add(new int[]{k, candidate});
                    }
                }
            }

            int[] chosen = bestCandidates.get(rnd.nextInt(bestCandidates.size()));
            trial[pos] = chosen[1];
            remaining.remove(chosen[0]);
        }

        return trial;
    }

    /** Picks the hole with the most already-known neighbour edges (ties broken randomly). */
    private int pickMostConstrainedHole(int[] trial, Set<Integer> remainingHoles, Random rnd) {
        int bestKnown = -1;
        List<Integer> tiedBest = new ArrayList<>();
        for (int pos : remainingHoles) {
            int row = pos / 16, col = pos % 16;
            int known = 0;
            if (row == 0  || isPlaced(trial[pos - 16])) known++;
            if (row == 15 || isPlaced(trial[pos + 16])) known++;
            if (col == 0  || isPlaced(trial[pos - 1]))  known++;
            if (col == 15 || isPlaced(trial[pos + 1]))  known++;

            if (known > bestKnown) {
                bestKnown = known;
                tiedBest.clear();
                tiedBest.add(pos);
            } else if (known == bestKnown) {
                tiedBest.add(pos);
            }
        }
        return tiedBest.get(rnd.nextInt(tiedBest.size()));
    }

    private boolean isPlaced(int p) {
        return p != -1 && p != -2;
    }

    // -----------------------------------------------------------------------
    // Core logic
    // -----------------------------------------------------------------------

    private int reduce(int[] board, int maxPasses, boolean verbose) {
        int before = countConflicts(board);
        if (before == 0) return 0;

        int current = before;
        for (int pass = 0; pass < maxPasses; pass++) {
            int afterRotation = rotationPass(board);
            int afterSwap     = swapPass(board);

//            if (verbose) {
//                logger.info(String.format(
//                        ">>> [CONFLICT REDUCER] Pass %d: %d → rot→%d → swap→%d",
//                        pass + 1, current, afterRotation, afterSwap));
//            }

            if (afterSwap >= current) break; // no improvement — stop early
            current = afterSwap;
        }

        int after = countConflicts(board);
//        if (after < before) {
//            logger.info(String.format(
//                    ">>> [CONFLICT REDUCER] %d → %d conflicts (-%d) on %d-piece board.",
//                    before, after, before - after, countPieces(board)));
//        }
        return after;
    }

    // -----------------------------------------------------------------------
    // Strategy 1: rotation pass
    // -----------------------------------------------------------------------

    private int rotationPass(int[] board) {
        int[] conflictScore = scoreConflicts(board);

        // Sort positions by conflict score descending — fix worst first
        List<Integer> conflicted = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            if (conflictScore[i] > 0) conflicted.add(i);
        }
        conflicted.sort((a, b) -> conflictScore[b] - conflictScore[a]);

        for (int pos : conflicted) {
            if (isLocked(pos)) continue;
            int current = board[pos];
            if (current == -1 || current == -2) continue;

            int bestPiece  = current;
            int bestConflicts = countConflicts(board);

            // Try all 4 orientations of the same physical piece
            int physId = getPhysId(current);
            if (physId < 0) continue;

            for (int oi = 0; oi < 1024; oi++) {
                if (inventory.physicalMapping[oi] != physId) continue;
                int candidate = inventory.allOrientations[oi];
                if (candidate == current) continue;

                board[pos] = candidate;
                int c = countConflicts(board);
                if (c < bestConflicts) {
                    bestConflicts = c;
                    bestPiece = candidate;
                }
            }
            board[pos] = bestPiece;
        }

        return countConflicts(board);
    }

    // -----------------------------------------------------------------------
    // Strategy 2: swap pass
    // -----------------------------------------------------------------------

    private int swapPass(int[] board) {
        int[] conflictScore = scoreConflicts(board);

        List<Integer> conflicted = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            if (conflictScore[i] > 0) conflicted.add(i);
        }

        for (int posA : conflicted) {
            if (isLocked(posA)) continue;
            int pA = board[posA];
            if (pA == -1 || pA == -2) continue;

            int baseline = countConflicts(board);

            for (int posB = 0; posB < 256; posB++) {
                if (posB == posA) continue;
                if (isLocked(posB)) continue;
                int pB = board[posB];
                if (pB == -1 || pB == -2) continue;

                // Try swap
                board[posA] = pB;
                board[posB] = pA;
                int afterSwap = countConflicts(board);

                if (afterSwap < baseline) {
                    // Keep the swap — also try all rotations of both pieces in
                    // their new positions to get extra mileage from the swap
                    baseline = afterSwap;
                    pA = board[posA]; // update for next iteration
                    pB = board[posB];
                } else {
                    // Revert
                    board[posA] = pA;
                    board[posB] = pB;
                }
            }
        }

        return countConflicts(board);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    public int countConflicts(int[] board) {
        int total = 0;
        for (int i = 0; i < 256; i++) {
            int p = board[i];
            if (p == -1 || p == -2) continue;
            int row = i / 16, col = i % 16;
            if (col < 15 && board[i+1] != -1 && board[i+1] != -2) {
                if (PieceUtils.getEast(p) != PieceUtils.getWest(board[i+1])) total++;
            }
            if (row < 15 && board[i+16] != -1 && board[i+16] != -2) {
                if (PieceUtils.getSouth(p) != PieceUtils.getNorth(board[i+16])) total++;
            }
            // Frame violations: a border-row/column piece whose outward edge
            // isn't grey is just as much a conflict as a mismatched neighbor,
            // but has no neighbor to mismatch against — must be checked
            // explicitly. Matches HoleSolver.findConflicts's definition;
            // omitting this let every heuristic below silently trade an
            // internal conflict for a new frame violation and report it as
            // an improvement.
            if (row == 0  && PieceUtils.getNorth(p) != PieceUtils.BORDER_COLOR) total++;
            if (row == 15 && PieceUtils.getSouth(p) != PieceUtils.BORDER_COLOR) total++;
            if (col == 0  && PieceUtils.getWest(p)  != PieceUtils.BORDER_COLOR) total++;
            if (col == 15 && PieceUtils.getEast(p)  != PieceUtils.BORDER_COLOR) total++;
        }
        return total;
    }

    private int[] scoreConflicts(int[] board) {
        int[] score = new int[256];
        for (int i = 0; i < 256; i++) {
            int p = board[i];
            if (p == -1 || p == -2) continue;
            int row = i / 16, col = i % 16;
            if (row > 0  && board[i-16] != -1 && PieceUtils.getNorth(p) != PieceUtils.getSouth(board[i-16])) score[i]++;
            if (row < 15 && board[i+16] != -1 && PieceUtils.getSouth(p) != PieceUtils.getNorth(board[i+16])) score[i]++;
            if (col > 0  && board[i-1]  != -1 && PieceUtils.getWest(p)  != PieceUtils.getEast(board[i-1]))  score[i]++;
            if (col < 15 && board[i+1]  != -1 && PieceUtils.getEast(p)  != PieceUtils.getWest(board[i+1]))  score[i]++;
            // Frame violations — see the matching comment in countConflicts.
            if (row == 0  && PieceUtils.getNorth(p) != PieceUtils.BORDER_COLOR) score[i]++;
            if (row == 15 && PieceUtils.getSouth(p) != PieceUtils.BORDER_COLOR) score[i]++;
            if (col == 0  && PieceUtils.getWest(p)  != PieceUtils.BORDER_COLOR) score[i]++;
            if (col == 15 && PieceUtils.getEast(p)  != PieceUtils.BORDER_COLOR) score[i]++;
        }
        return score;
    }

    private int countPieces(int[] board) {
        int n = 0;
        for (int p : board) if (p != -1 && p != -2) n++;
        return n;
    }

    private boolean isLocked(int pos) {
        // Center and hints are locked as a unit, matching how the rest of
        // the codebase treats lockCenter (see EternitySolver's static-lock
        // handling) — previously the hint positions were locked unconditionally
        // regardless of this flag, so a caller passing lockCenter=false (e.g.
        // HoleSolver analysing an arbitrary external board with no real
        // "hints" of its own) still had cells 221/45/210/34 wrongly excluded
        // from rotation/swap repair.
        if (!lockCenter) return false;
        if (pos == 135) return true;
        return pos == 221 || pos == 45 || pos == 210 || pos == 34;
    }

    private int getPhysId(int orientedPiece) {
        for (int oi = 0; oi < 1024; oi++) {
            if (inventory.allOrientations[oi] == orientedPiece) {
                return inventory.physicalMapping[oi];
            }
        }
        return -1;
    }
}