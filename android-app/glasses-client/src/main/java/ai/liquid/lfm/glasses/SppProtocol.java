package ai.liquid.lfm.glasses;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * LFM glasses SPP wire protocol (glasses side; mirrors the phone-side
 * Kotlin implementation).
 *
 * Frame: [0x80 | type:1][len:2 big-endian][payload:len]
 * - AUDIO_IN (0x01): glasses -> phone, mic PCM16 little-endian, 16 kHz mono.
 * - AUDIO_OUT (0x02): phone -> glasses, TTS PCM16 little-endian; the sample
 *   rate is announced by the TTS_START control message that precedes the
 *   frames of a response.
 * - CONTROL (0x03): UTF-8 JSON, both directions.
 */
final class SppProtocol {
    static final int AUDIO_IN = 0x01;
    static final int AUDIO_OUT = 0x02;
    static final int CONTROL = 0x03;
    private static final int FRAME_FLAG = 0x80;
    private static final int MAX_PAYLOAD = 256 * 1024;

    private SppProtocol() {
    }

    static void writeFrame(OutputStream out, int type, byte[] payload) throws IOException {
        byte[] frame = new byte[3 + payload.length];
        frame[0] = (byte) (FRAME_FLAG | type);
        frame[1] = (byte) ((payload.length >> 8) & 0xFF);
        frame[2] = (byte) (payload.length & 0xFF);
        System.arraycopy(payload, 0, frame, 3, payload.length);
        out.write(frame);
        out.flush();
    }

    static final class Frame {
        final int type;
        final byte[] payload;

        Frame(int type, byte[] payload) {
            this.type = type;
            this.payload = payload;
        }
    }

    static final class Reader {
        private final InputStream input;

        Reader(InputStream input) {
            this.input = input;
        }

        /** Returns null on a clean EOF. */
        Frame nextFrame() throws IOException {
            int header = input.read();
            if (header == -1) return null;
            if ((header & FRAME_FLAG) == 0) {
                throw new IOException("desync: bad frame flag 0x" + Integer.toHexString(header & 0xFF));
            }
            int type = header & 0x7F;
            int hi = input.read();
            int lo = input.read();
            if (hi == -1 || lo == -1) return null;
            int length = ((hi & 0xFF) << 8) | (lo & 0xFF);
            if (length > MAX_PAYLOAD) {
                throw new IOException("frame payload too large: " + length);
            }
            byte[] payload = new byte[length];
            int offset = 0;
            while (offset < length) {
                int read = input.read(payload, offset, length - offset);
                if (read == -1) throw new IOException("truncated frame payload");
                offset += read;
            }
            return new Frame(type, payload);
        }
    }
}
