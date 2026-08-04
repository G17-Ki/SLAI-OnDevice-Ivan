package ai.liquid.lfmdemo.tts

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal WAV reader/writer.
 *
 * Reading exists for ZipVoice, which is zero-shot and needs its prompt clip at
 * run time. Writing exists so a device-rendered utterance can be pulled back
 * and compared with the host golden byte for byte, which is the only way to
 * tell "the phone sounds different" from "the phone sounds the same and the
 * speaker is different".
 */
object WavIo {

    /** Returns (mono float samples in [-1,1], sample rate). Supports PCM16 and float32. */
    fun readMono(file: File): Pair<FloatArray, Int> {
        val bytes = file.readBytes()
        require(bytes.size > 44) { "wav too short: ${file.name}" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bb.int == 0x46464952) { "not RIFF: ${file.name}" }   // "RIFF"
        bb.int                                                        // riff size
        require(bb.int == 0x45564157) { "not WAVE: ${file.name}" }   // "WAVE"

        var audioFormat = 1
        var channels = 1
        var sampleRate = 0
        var bitsPerSample = 16
        var dataOffset = -1
        var dataSize = 0

        while (bb.remaining() >= 8) {
            val id = bb.int
            val size = bb.int
            val next = bb.position() + size + (size and 1)
            when (id) {
                0x20746D66 -> {                                       // "fmt "
                    audioFormat = bb.short.toInt() and 0xffff
                    channels = bb.short.toInt() and 0xffff
                    sampleRate = bb.int
                    bb.int                                            // byte rate
                    bb.short                                          // block align
                    bitsPerSample = bb.short.toInt() and 0xffff
                }
                0x61746164 -> {                                       // "data"
                    dataOffset = bb.position()
                    dataSize = size
                }
            }
            if (next > bb.capacity()) break
            bb.position(next)
        }
        require(dataOffset >= 0 && sampleRate > 0) { "no data/fmt chunk: ${file.name}" }

        val db = ByteBuffer.wrap(bytes, dataOffset, dataSize).order(ByteOrder.LITTLE_ENDIAN)
        val frames = when {
            audioFormat == 3 && bitsPerSample == 32 -> dataSize / 4 / channels
            bitsPerSample == 16 -> dataSize / 2 / channels
            else -> throw IllegalArgumentException(
                "unsupported wav format=$audioFormat bits=$bitsPerSample in ${file.name}"
            )
        }
        val out = FloatArray(frames)
        for (i in 0 until frames) {
            var acc = 0f
            for (ch in 0 until channels) {
                acc += if (audioFormat == 3) db.float else db.short / 32768f
            }
            out[i] = acc / channels          // downmix; the prompts are mono anyway
        }
        return out to sampleRate
    }

    fun writePcm16(file: File, samples: FloatArray, sampleRate: Int) {
        file.parentFile?.mkdirs()
        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(0)
            val dataBytes = samples.size * 2
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.putInt(0x46464952)                 // RIFF
            header.putInt(36 + dataBytes)
            header.putInt(0x45564157)                 // WAVE
            header.putInt(0x20746D66)                 // "fmt "
            header.putInt(16)
            header.putShort(1)                        // PCM
            header.putShort(1)                        // mono
            header.putInt(sampleRate)
            header.putInt(sampleRate * 2)
            header.putShort(2)
            header.putShort(16)
            header.putInt(0x61746164)                 // "data"
            header.putInt(dataBytes)
            raf.write(header.array())

            val body = ByteBuffer.allocate(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
            for (v in samples) {
                val clamped = when {
                    v.isNaN() || v.isInfinite() -> 0f      // never serialize invalid PCM
                    v > 1f -> 1f
                    v < -1f -> -1f
                    else -> v
                }
                body.putShort((clamped * 32767f).toInt().toShort())
            }
            raf.write(body.array())
        }
    }
}
