package ai.liquid.lfmdemo.whisperprobe;

/** JNI bridge for the verification-only bundled-WAV Whisper frontend. */
final class WhisperFrontendNative {
    static {
        System.loadLibrary("whisper_frontend");
    }

    private WhisperFrontendNative() {}

    static native byte[] compute(
            byte[] wav,
            int expectedPcmFrames,
            byte[] hann,
            byte[] melFilters
    );
}
