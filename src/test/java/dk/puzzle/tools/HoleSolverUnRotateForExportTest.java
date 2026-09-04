package dk.puzzle.tools;

import dk.puzzle.core.PieceLoader;
import dk.puzzle.model.PieceInventory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression cover for {@link HoleSolver#unRotateForExport}, added 2026-09-04 so a board found
 * under BlackwoodSolver's ROTATE_INSTANCE_DEGREES=180 hint search can genuinely satisfy the
 * five-clue regime after export, not just the open/starter-only track.
 *
 * <p>The fixture is a real board saved by the live 180-degree GPU run
 * (Errors21_Base248_220341_239, 21 conflicts before any of this). Chosen over a synthetic board
 * because the whole point is verifying behaviour against what the actual rotated search produces,
 * repair quirks included.
 *
 * <p><b>Known gap, not yet fixed:</b> hint181 does not reliably end up correctly placed even
 * after this fix -- checked against 5 separate live-run boards, all 5 landed 4/5 clues with 181
 * specifically the one that failed every time (right piece, wrong rotation in some, wrong cell in
 * others). Root cause not yet identified; tracked as a follow-up, not asserted here as passing.
 */
class HoleSolverUnRotateForExportTest {

    private static final String ROTATED_LINK =
            "https://e2.bucas.name/#puzzle=KnudHansen&board_w=16&board_h=16&board_edges="
                    + "acdaadgcabmdabgbabjbacvbaepcabteacrbacocaencacqeacpcabpcaepbaabedtcaovntmtrvgiwtuksivwvkpmiwttkmrqrtovjqntqvqvgtmkrvppjpssks"
                    + "dadscsbanplsrqvpwoiqsnqovjnnistjkgtrrtrgjoqtqnqogrinrmorujvmrijj"
                    + "dadibvealrwvnkvrillkqiklnjmittpjkkntrwpkqngwqrpnilirrlslwvwtjgwv"
                    + "dadgeibawgiivvlglwvvkrvwmniopmonnlompwnsgsjwpgmsigtgslgowoklkqmo"
                    + "baeqboeaiisolprivwppvwlwiliworglonjrnpgnjnmpnsnntgwsgjigkgrjmtlg"
                    + "eaetendasvpnrkhvppskllkpijjlgisjjvvigsgvmtrsnlktwmuliommrhmolsnh"
                    + "eaesdofapprohkrpsmwkkigmjpjistopvvitgouvriqokkgiuvtkmqlvmqgqntmq"
                    + "eadtfqfarluqrqwlwinqgqiijskqommsijjmurijqpqrgtnptlqtloglggkompqg"
                    + "daepfoeauisowquinouqiqookwhqmmrwjvomijnvqphjnigpqphignkpkolnqwso"
                    + "eadweueasrtuusgrujosovmjhnlvrkmnokvknhskhrphgtmrhjntklqjljmlsiuj"
                    + "dacielfathjlgimhowuimqhwlhuqmkqhvulksuvuploumtplnqgtqwtqmsowuvps"
                    + "cafvfwdajhhwmwuhushwhvjsuwovqlhwlwolvgwwolugplulghhltvphoknvprhk"
                    + "facrdufahwkuuwswhtrwjkqtounkhhruourhwokuujuouhsjhunhpmmunwlmhptw"
                    + "cadpftdaksntsphsrsjpqugsnvhurtkvrujtkjhuumhjsutmnrqumhmrlsuhthus"
                    + "dadhdcaanfachbafjfabgfafhcafkfacjbafhcabhbactfabqeafmfaeufafubafdaab";

    private static PieceInventory inventory;
    private static int[] rotatedBoard;

    @BeforeAll
    static void loadBoard() {
        inventory = new PieceInventory(PieceLoader.loadPieces());
        rotatedBoard = HoleSolver.decodeBoardAuto(ROTATED_LINK, inventory, false);
    }

    @AfterEach
    void restoreDefault() {
        HoleSolver.NON_CENTER_HINTS_ENABLED = false;
    }

    @Test
    void zeroDegreesIsIdentity() {
        assertArrayEquals(rotatedBoard, HoleSolver.unRotateForExport(rotatedBoard, 0));
    }

    @Test
    void fixtureBoardStartsAtTheKnownConflictCount() {
        assertEquals(21, HoleSolver.findConflicts(rotatedBoard).size(),
                "fixture should be the known 21-conflict live-run board");
    }

    @Test
    void unRotateAloneNeverIncreasesConflictsAndLeavesTheCenterPairEmpty() {
        int[] unrotated = HoleSolver.unRotateForExport(rotatedBoard, 180);

        int before = HoleSolver.findConflicts(rotatedBoard).size();
        int after = HoleSolver.findConflicts(unrotated).size();
        assertTrue(after <= before,
                "blanking the center pair can only remove conflicts (mathematically), never add them: "
                        + before + " -> " + after);

        // Center (139) at (8,7)=135 and its 180-degree swap partner (7,8)=120 must both be
        // blanked -- the caller's applyCluePins()+repair is what refills them correctly.
        assertEquals(-1, unrotated[8 * 16 + 7], "center's official cell must be left empty for applyCluePins to claim");
        assertEquals(-1, unrotated[7 * 16 + 8], "139's displaced neighbour cell must also be left empty");
    }

    @Test
    void afterUnRotateAndRepairTheCenterAndThreeOfFourCornerHintsAreSatisfied() {
        int[] unrotated = HoleSolver.unRotateForExport(rotatedBoard, 180);

        HoleSolver.NON_CENTER_HINTS_ENABLED = true;
        HoleSolver.ConflictSolveResult result = HoleSolver.solveConflicts(unrotated, inventory, false, 200);
        int[] repaired = result.bestBoard();

        assertEquals(orientedPieceFor(139, 270), repaired[135], "center (139) must land at its official cell/rotation");
        assertEquals(orientedPieceFor(208, 270), repaired[34], "hint208 must land at its official cell/rotation");
        assertEquals(orientedPieceFor(255, 270), repaired[45], "hint255 must land at its official cell/rotation");
        assertEquals(orientedPieceFor(249, 0), repaired[221], "hint249 must land at its official cell/rotation");
        // hint181 (cell 210, rotation 270) is the known, not-yet-fixed gap -- see class javadoc.
    }

    private static int orientedPieceFor(int physicalPiece, int rotationDegrees) {
        int physId = physicalPiece - 1;
        for (int oi = 0; oi < 1024; oi++) {
            if (inventory.physicalMapping[oi] == physId && (oi % 4) * 90 == rotationDegrees) {
                return inventory.allOrientations[oi];
            }
        }
        return -1;
    }
}
