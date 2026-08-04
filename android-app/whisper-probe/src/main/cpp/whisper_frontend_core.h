#ifndef LMF_WHISPER_FRONTEND_CORE_H_
#define LMF_WHISPER_FRONTEND_CORE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  WHISPER_FRONTEND_SAMPLE_RATE = 16000,
  WHISPER_FRONTEND_PADDED_SAMPLES = 480000,
  WHISPER_FRONTEND_FFT_SIZE = 400,
  WHISPER_FRONTEND_HOP_SIZE = 160,
  WHISPER_FRONTEND_BINS = 201,
  WHISPER_FRONTEND_MELS = 80,
  WHISPER_FRONTEND_FRAMES = 3000,
  WHISPER_FRONTEND_ELEMENTS = 240000,
  WHISPER_FRONTEND_OUTPUT_BYTES = 480000,
  WHISPER_FRONTEND_HANN_BYTES = 1600,
  WHISPER_FRONTEND_MEL_BYTES = 64320,
};

typedef struct whisper_frontend whisper_frontend;

whisper_frontend *whisper_frontend_create(char *error, size_t error_size);

void whisper_frontend_destroy(whisper_frontend *frontend);

/*
 * Computes QAI/Hugging Face-compatible [1,80,3000] UINT16 features.
 *
 * The WAV must be mono PCM16 at 16 kHz with exactly expected_pcm_frames.
 * hann_bytes is the pinned 400-value float32 little-endian periodic Hann
 * table; mel_bytes is the pinned [201,80] float32 little-endian matrix.
 * output_u16 receives WHISPER_FRONTEND_ELEMENTS native uint16 values in
 * [mel,frame] row-major order.
 */
int whisper_frontend_compute(
    whisper_frontend *frontend,
    const uint8_t *wav_bytes,
    size_t wav_size,
    uint32_t expected_pcm_frames,
    const uint8_t *hann_bytes,
    size_t hann_size,
    const uint8_t *mel_bytes,
    size_t mel_size,
    uint16_t *output_u16,
    size_t output_elements,
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif  // LMF_WHISPER_FRONTEND_CORE_H_
