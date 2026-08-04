package ai.liquid.lfmdemo.whisperprobe;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Decoder for Qualcomm's NUL-delimited Whisper base vocabulary. */
public final class VocabDecoder {
    public static final int BASE_ENTRY_COUNT = 50257;
    public static final int RAW_NUL_TOKEN_ID = 188;
    public static final int EMPTY_TOKEN_ID = 50256;
    public static final int FIRST_SPECIAL_TOKEN_ID = 50257;

    private final List<byte[]> entries;

    private VocabDecoder(List<byte[]> entries) {
        this.entries = entries;
    }

    public static VocabDecoder load(File vocabFile) throws IOException {
        byte[] encoded = Files.readAllBytes(vocabFile.toPath());
        List<byte[]> entries = new ArrayList<>(BASE_ENTRY_COUNT);
        int start = 0;
        for (int index = 0; index < encoded.length; index++) {
            if (encoded[index] == 0) {
                entries.add(Arrays.copyOfRange(encoded, start, index));
                start = index + 1;
            }
        }
        if (start != encoded.length) {
            throw new IOException("vocab.bin is missing its final NUL delimiter");
        }
        if (entries.size() != BASE_ENTRY_COUNT) {
            throw new IOException(
                    "vocab.bin has " + entries.size() + " entries; expected " + BASE_ENTRY_COUNT
            );
        }
        if (entries.get(RAW_NUL_TOKEN_ID).length != 0) {
            throw new IOException("vocab.bin token 188 is not the encoded raw-NUL field");
        }
        if (entries.get(EMPTY_TOKEN_ID).length != 0) {
            throw new IOException("vocab.bin token 50256 is not the expected empty token");
        }
        return new VocabDecoder(entries);
    }

    public String decode(List<Integer> tokenIds) {
        ByteArrayOutputStream joined = new ByteArrayOutputStream();
        for (int tokenId : tokenIds) {
            if (tokenId >= FIRST_SPECIAL_TOKEN_ID) {
                continue;
            }
            if (tokenId < 0 || tokenId >= entries.size()) {
                throw new IllegalArgumentException("token id outside base vocabulary: " + tokenId);
            }
            if (tokenId == RAW_NUL_TOKEN_ID) {
                joined.write(0);
            } else if (tokenId != EMPTY_TOKEN_ID) {
                byte[] token = entries.get(tokenId);
                joined.write(token, 0, token.length);
            }
        }
        // Token boundaries can split a multi-byte sequence. Decode only after
        // concatenating every raw token byte, and never accept replacement
        // characters for malformed model output.
        return decodeUtf8Strict(joined.toByteArray());
    }

    private static String decodeUtf8Strict(byte[] value) {
        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        final String decoded;
        try {
            decoded = decoder.decode(ByteBuffer.wrap(value)).toString();
        } catch (CharacterCodingException error) {
            throw new IllegalArgumentException(
                    "generated token bytes are not valid UTF-8", error
            );
        }
        for (int offset = 0; offset < decoded.length();) {
            int codePoint = decoded.codePointAt(offset);
            if (codePoint == 0) {
                throw new IllegalArgumentException(
                        "generated transcript contains a NUL character"
                );
            }
            if (codePoint == 0xfffd) {
                throw new IllegalArgumentException(
                        "generated transcript contains U+FFFD"
                );
            }
            if (Character.isISOControl(codePoint)
                    && codePoint != '\t'
                    && codePoint != '\n'
                    && codePoint != '\r') {
                throw new IllegalArgumentException(
                        String.format(
                                "generated transcript contains control U+%04X",
                                codePoint
                        )
                );
            }
            offset += Character.charCount(codePoint);
        }
        return decoded;
    }
}
