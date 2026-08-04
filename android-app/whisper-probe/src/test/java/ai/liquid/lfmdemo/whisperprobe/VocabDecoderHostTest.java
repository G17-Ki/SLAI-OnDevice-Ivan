package ai.liquid.lfmdemo.whisperprobe;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/** Dependency-free multilingual and fail-closed regression for VocabDecoder. */
public final class VocabDecoderHostTest {
    private VocabDecoderHostTest() {}

    public static void main(String[] args) throws Exception {
        Path temporary = Files.createTempFile("whisper-vocab-host-test-", ".bin");
        try {
            VocabDecoder decoder = VocabDecoder.load(writeFixtureVocab(temporary));
            require(
                    decoder.decode(List.of(
                            27908, 9254, 5975, 17174, 11249, 222, 47446,
                            30312, 10115, 235, 8861, 26668, 31375, 1543
                    )).equals("请用中文简单介绍一下北京。"),
                    "ZH reference vector"
            );
            require(
                    decoder.decode(List.of(
                            42334, 9143, 7785, 2785, 23455,
                            1098, 20879, 4215, 28517, 30
                    )).equals("프랑스의 수도는 어디인가요?"),
                    "KO reference vector"
            );
            require(
                    decoder.decode(List.of(
                            11179, 2019, 4218, 43862, 11384, 1631,
                            3752, 1000, 38842, 387, 13
                    )).equals("Расскажи коротко о Москве."),
                    "RU reference vector"
            );
            expectFailure(() -> decoder.decode(List.of(1)), "malformed UTF-8");
            expectFailure(() -> decoder.decode(List.of(2)), "U+FFFD");
            expectFailure(
                    () -> decoder.decode(List.of(VocabDecoder.RAW_NUL_TOKEN_ID)),
                    "NUL"
            );
            expectFailure(() -> decoder.decode(List.of(3)), "control character");
            require(decoder.decode(List.of(4)).equals("\t\n\r"), "allowed whitespace");
            require(
                    decoder.decode(List.of(27908, 50257, 9254)).equals("请用"),
                    "special tokens remain skipped"
            );
        } finally {
            Files.deleteIfExists(temporary);
        }
        System.out.println("VocabDecoderHostTest passed");
    }

    private static java.io.File writeFixtureVocab(Path path) throws Exception {
        byte[][] entries = new byte[VocabDecoder.BASE_ENTRY_COUNT][];
        for (int index = 0; index < entries.length; index++) {
            entries[index] = new byte[0];
        }

        putHex(entries, 1, "c3");
        putHex(entries, 2, "efbfbd");
        putHex(entries, 3, "01");
        entries[4] = "\t\n\r".getBytes(StandardCharsets.UTF_8);
        putHex(entries, 13, "2e");
        putHex(entries, 30, "3f");
        putHex(entries, 222, "80");
        putHex(entries, 235, "8d");
        putHex(entries, 387, "d0b5");
        putHex(entries, 1000, "20d0be");
        putHex(entries, 1098, "eb8a94");
        putHex(entries, 1543, "e38082");
        putHex(entries, 1631, "d0bed182");
        putHex(entries, 2019, "d0b0d181");
        putHex(entries, 2785, "ec9d98");
        putHex(entries, 3752, "d0bad0be");
        putHex(entries, 4215, "ec9db8");
        putHex(entries, 4218, "d181d0ba");
        putHex(entries, 5975, "e4b8ad");
        putHex(entries, 7785, "ec8aa4");
        putHex(entries, 8861, "e4b880e4b88b");
        putHex(entries, 9143, "9e91");
        putHex(entries, 9254, "e794a8");
        putHex(entries, 10115, "e7bb");
        putHex(entries, 11179, "d0a0");
        putHex(entries, 11249, "e7ae");
        putHex(entries, 11384, "20d0bad0bed180");
        putHex(entries, 17174, "e69687");
        putHex(entries, 20879, "20ec96b4eb9494");
        putHex(entries, 23455, "20ec8898eb8f84");
        putHex(entries, 26668, "e58c97");
        putHex(entries, 27908, "e8afb7");
        putHex(entries, 28517, "eab080ec9a94");
        putHex(entries, 30312, "e4bb8b");
        putHex(entries, 31375, "e4baac");
        putHex(entries, 38842, "20d09cd0bed181d0bad0b2");
        putHex(entries, 42334, "ed9484eb");
        putHex(entries, 43862, "d0b0d0b6d0b8");
        putHex(entries, 47446, "e58d95");

        ByteArrayOutputStream encoded = new ByteArrayOutputStream();
        for (byte[] entry : entries) {
            encoded.write(entry);
            encoded.write(0);
        }
        Files.write(path, encoded.toByteArray());
        return path.toFile();
    }

    private static void putHex(byte[][] entries, int tokenId, String hex) {
        byte[] value = new byte[hex.length() / 2];
        for (int index = 0; index < value.length; index++) {
            value[index] = (byte) Integer.parseInt(
                    hex.substring(index * 2, index * 2 + 2), 16
            );
        }
        entries[tokenId] = value;
    }

    private static void expectFailure(Runnable action, String label) {
        try {
            action.run();
            throw new AssertionError("expected rejection for " + label);
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
