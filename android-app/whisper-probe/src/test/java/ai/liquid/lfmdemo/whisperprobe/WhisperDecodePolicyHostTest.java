package ai.liquid.lfmdemo.whisperprobe;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Dependency-free host regression for the semantic decode policy. */
public final class WhisperDecodePolicyHostTest {
    private WhisperDecodePolicyHostTest() {}

    public static void main(String[] args) {
        testPinnedLanguageTokens();
        testForcedPromptAndBudget();
        testForcedThenGeneratedInputs();
        testCumulativeAttentionMask();
        testSpecialTokenSuppression();
        testEqualLogitTieUsesLowerTokenId();
        testSemanticTraceInvariant();
        testStrictBounds();
        System.out.println("WhisperDecodePolicyHostTest passed");
    }

    private static void testPinnedLanguageTokens() {
        require(WhisperDecodePolicy.Language.fromCode("EN").tokenId() == 50259, "EN token");
        require(WhisperDecodePolicy.Language.fromCode("zh").tokenId() == 50260, "ZH token");
        require(WhisperDecodePolicy.Language.fromCode("ru").tokenId() == 50263, "RU token");
        require(WhisperDecodePolicy.Language.fromCode("ko").tokenId() == 50264, "KO token");
        require(WhisperDecodePolicy.Language.fromCode("ja").tokenId() == 50266, "JA token");
    }

    private static void testForcedPromptAndBudget() {
        int[] prompt = WhisperDecodePolicy.forcedPrompt(
                WhisperDecodePolicy.Language.ENGLISH
        );
        require(
                Arrays.equals(prompt, new int[] {50258, 50259, 50359, 50363}),
                "forced EN prompt"
        );
        require(WhisperDecodePolicy.maximumGeneratedTokens(prompt.length) == 196, "budget");
        int sampled = 0;
        for (int step = 0; step < WhisperDecodePolicy.MAX_DECODER_CALLS; step++) {
            if (WhisperDecodePolicy.shouldSelectGeneratedToken(step, prompt.length)) {
                sampled++;
            }
        }
        require(sampled == 196, "sampled token count");
    }

    private static void testForcedThenGeneratedInputs() {
        int[] prompt = WhisperDecodePolicy.forcedPrompt(
                WhisperDecodePolicy.Language.ENGLISH
        );
        List<Integer> generated = new ArrayList<>();
        for (int step = 0; step < prompt.length; step++) {
            require(
                    WhisperDecodePolicy.inputTokenForStep(step, prompt, generated) == prompt[step],
                    "forced input at step " + step
            );
            if (WhisperDecodePolicy.shouldSelectGeneratedToken(step, prompt.length)) {
                generated.add(42);
            }
        }
        require(
                WhisperDecodePolicy.inputTokenForStep(4, prompt, generated) == 42,
                "first generated input"
        );
        generated.add(43);
        require(
                WhisperDecodePolicy.inputTokenForStep(5, prompt, generated) == 43,
                "second generated input"
        );
    }

    private static void testCumulativeAttentionMask() {
        assertMask(0, 199, 1);
        assertMask(1, 198, 2);
        assertMask(3, 196, 4);
        assertMask(198, 1, 199);
    }

    private static void assertMask(int step, int expectedFirst, int expectedCount) {
        short[] mask = WhisperDecodePolicy.cumulativeAttentionMask(step);
        require(mask.length == 200, "mask length");
        require(
                WhisperDecodePolicy.firstUnmaskedIndex(step) == expectedFirst,
                "first unmasked at step " + step
        );
        int count = 0;
        for (int index = 0; index < mask.length; index++) {
            int value = mask[index] & 0xffff;
            if (value == 65535) {
                require(index >= expectedFirst, "unmasked prefix leak");
                count++;
            } else {
                require(value == 0 && index < expectedFirst, "invalid masked value");
            }
        }
        require(count == expectedCount, "unmasked count at step " + step);
    }

    private static void testSpecialTokenSuppression() {
        ShortBuffer logits = logits();
        logits.put(42, (short) 60000);
        logits.put(WhisperDecodePolicy.EOT_TOKEN_ID, (short) 50000);
        logits.put(WhisperDecodePolicy.Language.ENGLISH.tokenId(), (short) 65535);
        logits.put(WhisperDecodePolicy.NO_TIMESTAMPS_TOKEN_ID, (short) 65534);
        logits.put(WhisperDecodePolicy.LAST_SUPPRESSED_TOKEN_ID, (short) 65533);
        require(WhisperDecodePolicy.selectAllowedToken(logits) == 42, "special suppression");

        logits.put(WhisperDecodePolicy.EOT_TOKEN_ID, (short) 65000);
        require(
                WhisperDecodePolicy.selectAllowedToken(logits)
                        == WhisperDecodePolicy.EOT_TOKEN_ID,
                "EOT remains allowed"
        );
        require(WhisperDecodePolicy.isEndOfTranscript(50257), "EOT predicate");
        require(!WhisperDecodePolicy.isAllowedGeneratedToken(50258), "SOT suppressed");
    }

    private static void testEqualLogitTieUsesLowerTokenId() {
        ShortBuffer logits = logits();
        logits.put(41, (short) 62000);
        logits.put(42, (short) 62000);
        logits.put(WhisperDecodePolicy.Language.ENGLISH.tokenId(), (short) 65535);
        require(
                WhisperDecodePolicy.selectAllowedToken(logits) == 41,
                "equal allowed logits must choose the lower token ID"
        );
    }

    private static void testSemanticTraceInvariant() {
        int[] prompt = WhisperDecodePolicy.forcedPrompt(
                WhisperDecodePolicy.Language.ENGLISH
        );
        List<Integer> generated = new ArrayList<>();
        List<Integer> firstUnmasked = new ArrayList<>();
        int decoderCalls = 0;
        for (int step = 0; step < WhisperDecodePolicy.MAX_DECODER_CALLS; step++) {
            firstUnmasked.add(WhisperDecodePolicy.firstUnmaskedIndex(step));
            decoderCalls++;
            if (WhisperDecodePolicy.shouldSelectGeneratedToken(step, prompt.length)) {
                generated.add(42);
            }
        }
        require(firstUnmasked.size() == decoderCalls, "attention trace length");
        require(firstUnmasked.get(0) == 199, "attention trace first value");
        require(firstUnmasked.get(firstUnmasked.size() - 1) == 1, "attention trace last value");
        for (int index = 1; index < firstUnmasked.size(); index++) {
            require(
                    firstUnmasked.get(index) == firstUnmasked.get(index - 1) - 1,
                    "attention trace must descend by one"
            );
        }
        require(
                decoderCalls == WhisperDecodePolicy.expectedDecoderCalls(
                        prompt.length,
                        generated.size()
                ),
                "decoder call/generated token invariant"
        );
    }

    private static ShortBuffer logits() {
        return ByteBuffer
                .allocateDirect(WhisperDecodePolicy.LOGIT_COUNT * 2)
                .order(ByteOrder.nativeOrder())
                .asShortBuffer();
    }

    private static void testStrictBounds() {
        expectFailure(() -> WhisperDecodePolicy.cumulativeAttentionMask(-1));
        expectFailure(() -> WhisperDecodePolicy.cumulativeAttentionMask(199));
        expectFailure(() -> WhisperDecodePolicy.maximumGeneratedTokens(0));
        expectFailure(() -> WhisperDecodePolicy.expectedDecoderCalls(4, 0));
        expectFailure(() -> WhisperDecodePolicy.expectedDecoderCalls(4, 197));
        expectFailure(() -> WhisperDecodePolicy.Language.fromCode("de"));
        expectFailure(() -> WhisperDecodePolicy.selectAllowedToken(
                ByteBuffer.allocateDirect(2).order(ByteOrder.nativeOrder()).asShortBuffer()
        ));
    }

    private static void expectFailure(Runnable action) {
        try {
            action.run();
            throw new AssertionError("expected failure");
        } catch (IllegalArgumentException expected) {
            // Expected fail-closed path.
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
