package dk.puzzle.blackwood;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Covers the depth-weighted seed sampling added 2026-08-30. The property that matters in production
 * is that repeated draws differ (a fixed pool is what caused restarts to re-mine the same
 * neighbourhoods) while deep boards still dominate.
 */
class BwSeedLoaderSamplingTest {

    private static final int MIN_DEPTH = 245;

    /** Depth is the only field sampling looks at; the rest just has to be non-null where used. */
    private static List<BwSeedLoader.Seed> pool(int perDepth, int minDepth, int maxDepth) {
        List<BwSeedLoader.Seed> seeds = new ArrayList<>();
        for (int depth = minDepth; depth <= maxDepth; depth++) {
            for (int i = 0; i < perDepth; i++) {
                seeds.add(new BwSeedLoader.Seed(
                        Path.of("d" + depth + "_" + i + ".txt"), depth, new int[256], null, -1));
            }
        }
        return seeds;
    }

    private static Set<Path> sourcesOf(List<BwSeedLoader.Seed> seeds) {
        Set<Path> paths = new HashSet<>();
        for (BwSeedLoader.Seed s : seeds) paths.add(s.source());
        return paths;
    }

    @Test
    void samplingReturnsExactlyMaxSeeds() {
        List<BwSeedLoader.Seed> picked =
                BwSeedLoader.sampleWeightedByDepth(pool(20, MIN_DEPTH, 253), MIN_DEPTH, 30, new Random(1));
        assertEquals(30, picked.size());
        assertEquals(30, sourcesOf(picked).size(), "sampling must be without replacement");
    }

    @Test
    void repeatedDrawsDiffer() {
        // The whole point of the change: two runs must not resume from an identical pool.
        List<BwSeedLoader.Seed> pool = pool(20, MIN_DEPTH, 253);
        Set<Path> first = sourcesOf(BwSeedLoader.sampleWeightedByDepth(pool, MIN_DEPTH, 30, new Random(1)));
        Set<Path> second = sourcesOf(BwSeedLoader.sampleWeightedByDepth(pool, MIN_DEPTH, 30, new Random(2)));
        assertTrue(first.size() == 30 && second.size() == 30);
        assertTrue(!first.equals(second), "two independent draws should not produce the identical pool");
    }

    @Test
    void deepBoardsAreStronglyFavoured() {
        // With bias exponent 3, a 253 is 9^3 = 729x likelier than a 245. Over many draws the mean
        // sampled depth must sit far above the pool's own mean, or the weighting is not working.
        List<BwSeedLoader.Seed> pool = pool(20, MIN_DEPTH, 253); // uniform 20 per depth, mean 249
        Random rand = new Random(42);
        long total = 0;
        int count = 0;
        for (int run = 0; run < 40; run++) {
            for (BwSeedLoader.Seed s : BwSeedLoader.sampleWeightedByDepth(pool, MIN_DEPTH, 20, rand)) {
                total += s.depth();
                count++;
            }
        }
        double meanSampledDepth = (double) total / count;
        assertTrue(meanSampledDepth > 251.0,
                "expected deep-biased sampling well above the pool mean of 249, got " + meanSampledDepth);
    }

    @Test
    void shallowBoardsStillGetDrawnSometimes() {
        // Not a strict top-K: the pool must stay reachable, or sampling buys no new territory.
        List<BwSeedLoader.Seed> pool = pool(20, MIN_DEPTH, 253);
        Random rand = new Random(7);
        int shallowSeen = 0;
        for (int run = 0; run < 200; run++) {
            for (BwSeedLoader.Seed s : BwSeedLoader.sampleWeightedByDepth(pool, MIN_DEPTH, 20, rand)) {
                if (s.depth() <= 249) shallowSeen++;
            }
        }
        assertTrue(shallowSeen > 0, "strict top-K behaviour: no board below the deepest tier was ever drawn");
    }

    @Test
    void poolSmallerThanRequestIsReturnedWhole() {
        List<BwSeedLoader.Seed> pool = pool(2, MIN_DEPTH, 246); // 4 seeds
        List<BwSeedLoader.Seed> picked =
                BwSeedLoader.sampleWeightedByDepth(pool, MIN_DEPTH, 4, new Random(3));
        assertEquals(4, picked.size());
    }
}
