package ai.liquid.lfmdemo.whisperprobe;

import java.nio.ShortBuffer;
import java.util.List;
import java.util.Locale;

/**
 * Pure-Java, fail-closed policy for the fixed-shape Qualcomm Whisper decoder.
 *
 * <p>The decoder accepts one input token per call and has a 200-slot attention mask with a
 * 199-token self-cache. Calls therefore use positions 0..198. A four-token forced prompt leaves
 * 196 logits that may become generated transcript tokens.</p>
 */
public final class WhisperDecodePolicy {
    public static final int DECODER_CONTEXT_LENGTH = 200;
    public static final int MAX_DECODER_CALLS = DECODER_CONTEXT_LENGTH - 1;
    public static final int MAX_DECODER_POSITION = MAX_DECODER_CALLS - 1;
    public static final int LOGIT_COUNT = 51865;

    public static final int EOT_TOKEN_ID = 50257;
    public static final int SOT_TOKEN_ID = 50258;
    public static final int TRANSCRIBE_TOKEN_ID = 50359;
    public static final int NO_TIMESTAMPS_TOKEN_ID = 50363;
    public static final int LAST_BASE_TEXT_TOKEN_ID = 50256;
    public static final int FIRST_SUPPRESSED_TOKEN_ID = 50258;
    public static final int LAST_SUPPRESSED_TOKEN_ID = LOGIT_COUNT - 1;

    private WhisperDecodePolicy() {}

    /** Core demo languages, pinned to OpenAI Whisper multilingual special-token IDs. */
    public enum Language {
        ENGLISH("en", 50259),
        CHINESE("zh", 50260),
        RUSSIAN("ru", 50263),
        KOREAN("ko", 50264),
        JAPANESE("ja", 50266);

        private final String code;
        private final int tokenId;

        Language(String code, int tokenId) {
            this.code = code;
            this.tokenId = tokenId;
        }

        public String code() {
            return code;
        }

        public int tokenId() {
            return tokenId;
        }

        public static Language fromCode(String code) {
            if (code == null) {
                throw new IllegalArgumentException("language code is required");
            }
            String normalized = code.toLowerCase(Locale.ROOT);
            for (Language language : values()) {
                if (language.code.equals(normalized)) {
                    return language;
                }
            }
            throw new IllegalArgumentException("unsupported Whisper language: " + code);
        }
    }

    public static int[] forcedPrompt(Language language) {
        if (language == null) {
            throw new IllegalArgumentException("language is required");
        }
        return new int[] {
                SOT_TOKEN_ID,
                language.tokenId(),
                TRANSCRIBE_TOKEN_ID,
                NO_TIMESTAMPS_TOKEN_ID
        };
    }

    public static int maximumGeneratedTokens(int promptLength) {
        if (promptLength <= 0 || promptLength > MAX_DECODER_CALLS) {
            throw new IllegalArgumentException("invalid forced-prompt length: " + promptLength);
        }
        return DECODER_CONTEXT_LENGTH - promptLength;
    }

    public static int expectedDecoderCalls(int promptLength, int generatedTokenCount) {
        int maximumGenerated = maximumGeneratedTokens(promptLength);
        if (generatedTokenCount <= 0 || generatedTokenCount > maximumGenerated) {
            throw new IllegalArgumentException(
                    "invalid generated-token count: " + generatedTokenCount
            );
        }
        return promptLength - 1 + generatedTokenCount;
    }

    public static boolean shouldSelectGeneratedToken(int decoderStep, int promptLength) {
        requireDecoderStep(decoderStep);
        if (promptLength <= 0 || promptLength > MAX_DECODER_CALLS) {
            throw new IllegalArgumentException("invalid forced-prompt length: " + promptLength);
        }
        return decoderStep >= promptLength - 1;
    }

    public static int inputTokenForStep(
            int decoderStep,
            int[] prompt,
            List<Integer> generatedTokens
    ) {
        requireDecoderStep(decoderStep);
        if (prompt == null || prompt.length == 0) {
            throw new IllegalArgumentException("forced prompt is required");
        }
        if (generatedTokens == null) {
            throw new IllegalArgumentException("generated token list is required");
        }
        if (decoderStep < prompt.length) {
            return prompt[decoderStep];
        }
        int generatedIndex = decoderStep - prompt.length;
        if (generatedIndex >= generatedTokens.size()) {
            throw new IllegalStateException(
                    "missing generated token " + generatedIndex + " for decoder step " + decoderStep
            );
        }
        return generatedTokens.get(generatedIndex);
    }

    /** First unmasked slot for Qualcomm's cumulative right-aligned attention mask. */
    public static int firstUnmaskedIndex(int decoderStep) {
        requireDecoderStep(decoderStep);
        return DECODER_CONTEXT_LENGTH - decoderStep - 1;
    }

    /**
     * Returns the exact UINT16 mask payload: masked slots are 0 and the cumulative suffix is
     * 65535, the quantized representation of real attention bias 0 for this model.
     */
    public static short[] cumulativeAttentionMask(int decoderStep) {
        int firstUnmasked = firstUnmaskedIndex(decoderStep);
        short[] mask = new short[DECODER_CONTEXT_LENGTH];
        for (int index = firstUnmasked; index < mask.length; index++) {
            mask[index] = (short) 0xffff;
        }
        return mask;
    }

    /**
     * Greedy selection over transcript text IDs plus EOT only. All language, task, control, and
     * timestamp IDs 50258..51864 are suppressed by never entering the argmax domain.
     */
    public static int selectAllowedToken(ShortBuffer uint16Logits) {
        if (uint16Logits == null) {
            throw new IllegalArgumentException("logits are required");
        }
        ShortBuffer values = uint16Logits.duplicate();
        if (values.remaining() != LOGIT_COUNT) {
            throw new IllegalArgumentException(
                    "logits has " + values.remaining() + " values; expected " + LOGIT_COUNT
            );
        }
        int start = values.position();
        int bestIndex = 0;
        int bestValue = -1;
        for (int tokenId = 0; tokenId <= EOT_TOKEN_ID; tokenId++) {
            int value = values.get(start + tokenId) & 0xffff;
            if (value > bestValue) {
                bestValue = value;
                bestIndex = tokenId;
            }
        }
        return bestIndex;
    }

    public static boolean isEndOfTranscript(int tokenId) {
        return tokenId == EOT_TOKEN_ID;
    }

    public static boolean isAllowedGeneratedToken(int tokenId) {
        return tokenId >= 0 && tokenId <= EOT_TOKEN_ID;
    }

    private static void requireDecoderStep(int decoderStep) {
        if (decoderStep < 0 || decoderStep > MAX_DECODER_POSITION) {
            throw new IllegalArgumentException("decoder step outside 0..198: " + decoderStep);
        }
    }
}
