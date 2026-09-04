package dk.puzzle.tools;

import dk.puzzle.core.PieceLoader;
import dk.puzzle.model.PieceInventory;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Regression cover for {@link HoleSolver#rotateBoardCw}, added 2026-09-04 to support
 * BlackwoodSolver's ROTATE_INSTANCE_DEGREES experiment: a board found under a rotated hint
 * search has to be rotated back to the true official clue positions/orientations before it
 * means anything to HoleSolver's clue pins or bucas export.
 *
 * <p>The fixture is the same real, complete, bucas-verified 19-conflict board used by
 * {@link HoleSolverCluePinsTest}, reused here purely as "a valid board with a known conflict
 * count" -- its clue-compliance is irrelevant to what this test checks.
 */
class HoleSolverBoardRotationTest {

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

    @Test
    void zeroDegreesIsIdentity() {
        assertArrayEquals(validBoard, HoleSolver.rotateBoardCw(validBoard, 0));
    }

    @Test
    void everyQuarterTurnPreservesTheConflictCount() {
        int baseline = HoleSolver.findConflicts(validBoard).size();
        assertEquals(19, baseline, "fixture should be the known 19-conflict board");
        for (int degrees : new int[]{90, 180, 270}) {
            int[] rotated = HoleSolver.rotateBoardCw(validBoard, degrees);
            assertEquals(256, rotated.length);
            assertEquals(baseline, HoleSolver.findConflicts(rotated).size(),
                    "rotating by " + degrees + " must not change the conflict count");
        }
    }

    @Test
    void fourQuarterTurnsReturnToTheOriginalBoard() {
        int[] rotated = validBoard;
        for (int i = 0; i < 4; i++) {
            rotated = HoleSolver.rotateBoardCw(rotated, 90);
        }
        assertArrayEquals(validBoard, rotated, "four 90-degree turns must be the identity");
    }

    @Test
    void rotate90ThenRotate270IsTheIdentity() {
        int[] rotated = HoleSolver.rotateBoardCw(validBoard, 90);
        int[] back = HoleSolver.rotateBoardCw(rotated, 270);
        assertArrayEquals(validBoard, back,
                "rotating forward then back by the complementary amount must round-trip exactly");
    }

    @Test
    void rejectsNonMultipleOf90() {
        assertThrows(IllegalArgumentException.class, () -> HoleSolver.rotateBoardCw(validBoard, 45));
    }
}
