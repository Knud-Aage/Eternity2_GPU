package dk.puzzle.tools;

import dk.puzzle.core.PieceLoader;
import dk.puzzle.model.PieceInventory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression cover for the 2026-09-04 clue-pin fix in {@link HoleSolver}.
 *
 * <p>Background: 10 of the first 11 boards saved by the hints-enabled solvers had an
 * official clue piece sitting somewhere other than its pinned cell -- almost always 255,
 * once 208. The solver had placed every clue correctly; HoleSolver's repair passes moved
 * them afterwards, which silently disqualifies the board from the five-clue regime. Two
 * routes caused it, and this class pins both shut:
 *
 * <ol>
 *   <li>the clue cell is still EMPTY when the solver stops (at depth 247 cell 45 is
 *       unfilled), so MCV refill fills it with whatever scores best; and</li>
 *   <li>the clue cell gets swallowed by a conflict region, so RegionSolver re-places it.</li>
 * </ol>
 *
 * <p>The board used here is the one five-clue-valid board the old code happened to
 * produce (Errors19_Base247_204146_873, 19 conflicts = 461/480, confirmed by bucas's own
 * scorer). Because it starts compliant, any clue that has moved by the end of a run is
 * unambiguously HoleSolver's doing rather than an artefact of the input.</p>
 */
class HoleSolverCluePinsTest {

    /** Cell indices are top-down here (row 0 = north), matching HoleSolver's own indexing. */
    private static final int CELL_139 = 135, CELL_208 = 34, CELL_255 = 45, CELL_181 = 210, CELL_249 = 221;
    private static final int[] CLUE_CELLS = {CELL_139, CELL_208, CELL_255, CELL_181, CELL_249};

    private static final String VALID_461_LINK =
            "https://e2.bucas.name/#puzzle=KnudHansen&board_w=16&board_h=16&board_edges="
                    + "adcaadsdabmdaeqbacqeadpcadgdaepdaboeaepbabteacvbaepcaeteadweaacdcidasoiimnloqgtnsjwgpjijgmpqpgnnoslgpwnstwvwvgtqvvlgttkmqosn"
                    + "cacodidaiwgilmnwtlgmwollisoupgmsnkpglqiknqoqvntqtkknlwokswuwinqw"
                    + "cabpdteagigtnjmigisjllkiorglmorrpstogvgsoknvtnlkkvrnovmjvlwvqjkl"
                    + "babjendagwqnmsowsskskppsgtnprqrttjoqqovjnpmollkprlslmtplwlrqogll"
                    + "babgdgcaqiigomnikrvmpnqrsvpnrkhvolnkvmqlmrwmkmnrsommpprorilpliwi"
                    + "baeicrbaijjrnvijvitvqwoippvwhsspntksqqwtwsoqnplsmmuprmhmlljmwvwl"
                    + "eabvbjfajlijiriltrgronjrvjnnskqjkgikwwvgovuwlrwvusgrhvjsjgwvwstg"
                    + "bacsfgfaimhgiommggkojiggngriqmqgijjmvomjuvgowkrvgtrkjistwtgitpjt"
                    + "cacpfwdahqkwmokqkuwogoluriqoqphijpppmiwpgmkirgtmrjkgsiujgpnijnmp"
                    + "caendtcakvrtkokvwuiolkvuqtjkhjltprsjwhtrkprhtwhpksmwuvpsntovmsut"
                    + "eaesckfarwpkkuhwiwquvkvwjhuklghhsqugttlqrsmthwusmqhwpqrqooiquqno"
                    + "eafqfqfaphjqhwjhqlhwvhnlunhhhlsnuqrllhuqmorhuplohtvprvmtijvvnthj"
                    + "fabtfhcajumhjvmuhunvnrquhoursujorijuuksirphklulpvtkumqntvprqhhrp"
                    + "bafhchbamwuhmulwnkouqhmkusthjuhsjuousuvuhhrulsuhknhsnsnnrtusrujt"
                    + "fadubeaaueaelfaeoeafmfaetdafhdadofadvcafrfacubafhcabnfacufafjbafdaab";

    private static PieceInventory inventory;
    private static int[] validBoard;

    @BeforeAll
    static void loadBoard() {
        inventory = new PieceInventory(PieceLoader.loadPieces());
        validBoard = HoleSolver.decodeBoardAuto(VALID_461_LINK, inventory, false);
    }

    @AfterEach
    void restoreDefault() {
        HoleSolver.NON_CENTER_HINTS_ENABLED = false;
    }

    /** Sanity: the fixture really is a compliant five-clue board, or nothing below means anything. */
    @Test
    void fixtureBoardStartsWithAllFiveCluesInPlace() {
        assertEquals(256, validBoard.length);
        for (int cell : CLUE_CELLS) {
            assertNotEquals(-1, validBoard[cell], "clue cell " + cell + " should be occupied in the fixture");
        }
        assertEquals(19, HoleSolver.findConflicts(validBoard).size(),
                "fixture should be the known 19-conflict (461/480) board");
    }

    @Test
    void emptyClueCellIsRefilledWithItsOwnCluePieceNotWhateverScoresBest() {
        // Reproduces the depth-247 shape: cell 45 (piece 255) never reached by the solver.
        // Before the fix, MCV refill put an arbitrary piece here.
        HoleSolver.NON_CENTER_HINTS_ENABLED = true;
        int[] board = Arrays.copyOf(validBoard, 256);
        int expected255 = board[CELL_255];
        board[CELL_255] = -1;

        HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(board, inventory, false, 200);

        assertEquals(expected255, result.bestBoard()[CELL_255],
                "piece 255 must be restored to cell 45 at its official rotation");
    }

    @Test
    void allFiveCluesSurviveRepairWithHintsOn() {
        HoleSolver.NON_CENTER_HINTS_ENABLED = true;
        int[] board = Arrays.copyOf(validBoard, 256);
        // Blank all five clue cells plus a couple of neighbours, forcing both the region
        // path and the refill path to run right across every pinned cell at once.
        for (int cell : CLUE_CELLS) board[cell] = -1;
        board[CELL_255 + 1] = -1;
        board[CELL_208 + 1] = -1;

        HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(board, inventory, false, 200);
        int[] out = result.bestBoard();

        for (int cell : CLUE_CELLS) {
            assertEquals(validBoard[cell], out[cell],
                    "clue cell " + cell + " must hold its official piece and rotation after repair");
        }
    }

    @Test
    void hintsOffLeavesTheOldBehaviourAlone() {
        // The switch must not change no-hints or arbitrary-external-board runs: with it off,
        // nothing is pinned and a blanked clue cell is just an ordinary hole to fill.
        HoleSolver.NON_CENTER_HINTS_ENABLED = false;
        int[] board = Arrays.copyOf(validBoard, 256);
        board[CELL_255] = -1;

        HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(board, inventory, false, 50);

        assertNotEquals(-1, result.bestBoard()[CELL_255], "the hole should still get filled with hints off");
    }

    @Test
    void solveConflictsDoesNotMutateTheCallersBoard() {
        // applyCluePins writes a clue piece into an empty cell, so the defensive copy in
        // solveConflicts is load-bearing -- BlackwoodGpuRunner calls this in-process.
        HoleSolver.NON_CENTER_HINTS_ENABLED = true;
        int[] board = Arrays.copyOf(validBoard, 256);
        board[CELL_255] = -1;
        int[] before = Arrays.copyOf(board, 256);

        HoleSolver.solveConflicts(board, inventory, false, 50);

        assertTrue(Arrays.equals(before, board), "solveConflicts must not modify the array it was handed");
    }
}
