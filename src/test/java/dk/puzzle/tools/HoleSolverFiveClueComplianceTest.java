package dk.puzzle.tools;

import dk.puzzle.core.PieceLoader;
import dk.puzzle.model.PieceInventory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression cover for {@link HoleSolver#checkFiveClueCompliance}, added 2026-09-05 so the live
 * pipeline can report ground-truth five-clue compliance on every evaluated board (not just ones
 * a person happens to spot-check), after finding that manual offline checks and the live,
 * post-restart run disagreed on how reliable hint181 was.
 */
class HoleSolverFiveClueComplianceTest {

    /** Real board saved by the live 180-degree run after the unRotateForExport fix, confirmed 5/5 by hand. */
    private static final String FULLY_COMPLIANT_LINK =
            "https://e2.bucas.name/#puzzle=KnudHansen&board_w=16&board_h=16&board_edges="
                    + "abdaafubafufadufaftdabtfafjbacrfafhcaemfaeueafqeafgfadofadhdaacddweaushwuhlsunhhthjntvphjshvrtushustmhjuunvhqrpngtmropsthhrp"
                    + "cabhelfahwqllmnwhwmqjhhwptwhhjltuhsjsknhjhukvrkhphkrmwuhswuwrhou"
                    + "bafhfwdaquiwnrqumhmrhruhwhtrlghhsqugnouquisoklqikvulusuvujosoqtj"
                    + "fafqdgcaiiwgqphimonpujuotrujhmorulwmulplsnhlqosnuvgouwovokuwtnlk"
                    + "facncocawmsohgimnkpguhwkuqlhoriqwpkrppjphjqpsiujggjioluguqrllvmq"
                    + "cafvchbassphiuksplouwvwllrwvilprkplljijpqooiunkojminupmmrhkpmkqh"
                    + "fackbmdapqgmkwhqoiqwwiliwtgiplmtlkilpmiworrmkgtrisjgmwkskrvwqpqr"
                    + "caepdsdagvgshnlvqgtnlloggoslmmsoijjmijurrijjtvvijgwvkoggvntoqoqn"
                    + "eafodteagwstlrqwtrgropprsuvpsgruwwvguiowjjlivvijwppvgtnptkmtqjsk"
                    + "fabjeibasoiiqwsogsjwpgmsvvlgrnkvvjnnovmjlwvvinqwplsnnkolmnrkspwn"
                    + "baepboeavjqoklqjjmllmiomlirikkginiommsutvpnspskpskssokvkrvmkwvkv"
                    + "eabveseaqttlqiiglwokollwrlslgmtlokqmuvtkntqvkkntsntkvoknmjvokgrj"
                    + "babgepdarsmtjvmuonjrlomnrglotqvgqwtqtwvwqngwnsnntjiskqtjvprqrsjp"
                    + "bacsdgdammrwmpjnjttpmqntmkigvrtktrqrvmtrqgqmnnpgigpntgigringjnvi"
                    + "caendcaarbacjbabteabndaeidadtcadqeacteaeqbaepcabpcacidacpcadvbaceaab";

    private static final String ROTATED_LINK_STILL_MISSING_181 =
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

    @BeforeAll
    static void loadInventory() {
        inventory = new PieceInventory(PieceLoader.loadPieces());
    }

    @AfterEach
    void restoreDefault() {
        HoleSolver.NON_CENTER_HINTS_ENABLED = false;
    }

    @Test
    void fullyCompliantBoardReportsFiveOfFive() {
        int[] board = HoleSolver.decodeBoardAuto(FULLY_COMPLIANT_LINK, inventory, false);
        HoleSolver.FiveClueCompliance result = HoleSolver.checkFiveClueCompliance(board, inventory);
        assertTrue(result.isFullyCompliant());
        assertEquals(5, result.matchedCount());
        assertTrue(result.failedPieces().isEmpty());
    }

    @Test
    void partiallyCompliantBoardNamesTheFailedPiece() {
        int[] board = HoleSolver.decodeBoardAuto(ROTATED_LINK_STILL_MISSING_181, inventory, false);
        int[] unrotated = HoleSolver.unRotateForExport(board, 180);

        HoleSolver.NON_CENTER_HINTS_ENABLED = true;
        HoleSolver.ConflictSolveResult repairResult = HoleSolver.solveConflicts(unrotated, inventory, false, 200);

        HoleSolver.FiveClueCompliance result = HoleSolver.checkFiveClueCompliance(repairResult.bestBoard(), inventory);
        assertFalse(result.isFullyCompliant());
        assertEquals(4, result.matchedCount());
        assertEquals(1, result.failedPieces().size());
        assertEquals(181, result.failedPieces().get(0));
    }
}
