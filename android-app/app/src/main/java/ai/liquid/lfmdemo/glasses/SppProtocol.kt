package ai.liquid.lfmdemo.glasses

import java.io.IOException
import java.io.InputStream

/**
 * LFM glasses SPP wire protocol.
 *
 * Frame: [0x80 | type:1][len:2 big-endian][payload:len]
 * - AUDIO_IN (0x01): glasses -> phone, mic PCM16 little-endian, 16 kHz mono.
 * - AUDIO_OUT (0x02): phone -> glasses, TTS PCM16 little-endian; the sample
 *   rate is announced by the TTS_START control message that precedes the
 *   frames of a response.
 * - CONTROL (0x03): UTF-8 JSON, both directions.
 */
object SppProtocol {
    const val AUDIO_IN = 0x01
    const val AUDIO_OUT = 0x02
    const val CONTROL = 0x03
    private const val FRAME_FLAG = 0x80
    private const val MAX_PAYLOAD = 256 * 1024

    fun encode(type: Int, payload: ByteArray): ByteArray {
        require(type in 0..0x7F) { "type must fit in the low 7 bits" }
        require(payload.size <= MAX_PAYLOAD) { "payload too large: ${payload.size}" }
        val out = ByteArray(3 + payload.size)
        out[0] = (FRAME_FLAG or type).toByte()
        out[1] = ((payload.size shr 8) and 0xFF).toByte()
        out[2] = (payload.size and 0xFF).toByte()
        System.arraycopy(payload, 0, out, 3, payload.size)
        return out
    }

    class Reader(private val input: InputStream) {
        fun nextFrame(): Frame? {
            val header = input.read()
            if (header == -1) return null
            if ((header and FRAME_FLAG) == 0) {
                throw IOException("desync: bad frame flag 0x${(header and 0xFF).toString(16)}")
            }
            val type = header and 0x7F
            val hi = input.read()
            val lo = input.read()
            if (hi == -1 || lo == -1) return null
            val length = ((hi and 0xFF) shl 8) or (lo and 0xFF)
            if (length > MAX_PAYLOAD) {
                throw IOException("frame payload too large: $length")
            }
            val payload = ByteArray(length)
            var offset = 0
            while (offset < length) {
                val read = input.read(payload, offset, length - offset)
                if (read == -1) throw IOException("truncated frame payload")
                offset += read
            }
            return Frame(type, payload)
        }
    }

    data class Frame(val type: Int, val payload: ByteArray)
}
