#include "whisper_frontend_core.h"

#include <fenv.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kiss_fftr.h"

#pragma STDC FENV_ACCESS ON

static const double kFeatureQuantizationScale = 4.677007018472068e-05;
static const int32_t kFeatureZeroPoint = 32072;

struct whisper_frontend {
  kiss_fftr_cfg fft;
};

static int fail(char *error, size_t error_size, const char *format, ...) {
  if (error != NULL && error_size > 0) {
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
    error[error_size - 1] = '\0';
  }
  return 0;
}

static uint16_t read_u16_le(const uint8_t *value) {
  return (uint16_t)((uint16_t)value[0] | ((uint16_t)value[1] << 8));
}

static uint32_t read_u32_le(const uint8_t *value) {
  return (uint32_t)value[0] | ((uint32_t)value[1] << 8) |
         ((uint32_t)value[2] << 16) | ((uint32_t)value[3] << 24);
}

static float read_f32_le(const uint8_t *value) {
  const uint32_t bits = read_u32_le(value);
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static int32_t torch_reflect_index(int64_t index, int32_t size) {
  while (index < 0 || index >= size) {
    if (index < 0) {
      /* torch.nn.functional.pad(..., mode="reflect"): -1 -> 1. */
      index = -index;
    } else {
      index = 2LL * size - 2 - index;
    }
  }
  return (int32_t)index;
}

typedef struct {
  const uint8_t *samples;
  uint32_t frames;
} pcm_view;

static int parse_pcm16_wav(const uint8_t *wav,
                           size_t wav_size,
                           uint32_t expected_frames,
                           pcm_view *view,
                           char *error,
                           size_t error_size) {
  if (wav == NULL || view == NULL || wav_size < 44) {
    return fail(error, error_size, "WAV is null or truncated");
  }
  if (memcmp(wav, "RIFF", 4) != 0 || memcmp(wav + 8, "WAVE", 4) != 0) {
    return fail(error, error_size, "WAV must be RIFF/WAVE");
  }
  const uint64_t declared_size = (uint64_t)read_u32_le(wav + 4) + 8U;
  if (declared_size != wav_size) {
    return fail(error, error_size, "WAV RIFF size mismatch");
  }

  int found_format = 0;
  int found_data = 0;
  uint16_t audio_format = 0;
  uint16_t channels = 0;
  uint16_t block_align = 0;
  uint16_t bits_per_sample = 0;
  uint32_t sample_rate = 0;
  uint32_t byte_rate = 0;
  const uint8_t *pcm = NULL;
  uint32_t pcm_bytes = 0;

  size_t offset = 12;
  while (offset + 8 <= wav_size) {
    const uint8_t *chunk = wav + offset;
    const uint32_t chunk_size = read_u32_le(chunk + 4);
    const uint64_t data_end = (uint64_t)offset + 8U + chunk_size;
    if (data_end > wav_size) {
      return fail(error, error_size, "WAV chunk exceeds file bounds");
    }
    if (memcmp(chunk, "fmt ", 4) == 0) {
      if (found_format || chunk_size < 16) {
        return fail(error, error_size, "WAV has invalid fmt chunk");
      }
      const uint8_t *format = chunk + 8;
      audio_format = read_u16_le(format);
      channels = read_u16_le(format + 2);
      sample_rate = read_u32_le(format + 4);
      byte_rate = read_u32_le(format + 8);
      block_align = read_u16_le(format + 12);
      bits_per_sample = read_u16_le(format + 14);
      found_format = 1;
    } else if (memcmp(chunk, "data", 4) == 0) {
      if (found_data) {
        return fail(error, error_size, "WAV has multiple data chunks");
      }
      pcm = chunk + 8;
      pcm_bytes = chunk_size;
      found_data = 1;
    }
    offset = (size_t)data_end + (chunk_size & 1U);
  }
  if (offset != wav_size || !found_format || !found_data) {
    return fail(error, error_size, "WAV is missing a complete fmt/data layout");
  }
  if (audio_format != 1 || channels != 1 || sample_rate != 16000 ||
      byte_rate != 32000 || block_align != 2 || bits_per_sample != 16) {
    return fail(error, error_size, "WAV must be mono PCM16 at 16000 Hz");
  }
  if ((pcm_bytes & 1U) != 0 || pcm_bytes / 2U != expected_frames) {
    return fail(error, error_size, "WAV PCM frame count mismatch");
  }
  if (expected_frames > WHISPER_FRONTEND_PADDED_SAMPLES) {
    return fail(error, error_size, "WAV exceeds the 30-second frontend window");
  }
  view->samples = pcm;
  view->frames = expected_frames;
  return 1;
}

whisper_frontend *whisper_frontend_create(char *error, size_t error_size) {
  whisper_frontend *frontend = (whisper_frontend *)calloc(1, sizeof(*frontend));
  if (frontend == NULL) {
    fail(error, error_size, "cannot allocate frontend");
    return NULL;
  }
  frontend->fft = kiss_fftr_alloc(WHISPER_FRONTEND_FFT_SIZE, 0, NULL, NULL);
  if (frontend->fft == NULL) {
    free(frontend);
    fail(error, error_size, "kiss_fftr_alloc(400) failed");
    return NULL;
  }
  return frontend;
}

void whisper_frontend_destroy(whisper_frontend *frontend) {
  if (frontend != NULL) {
    if (frontend->fft != NULL) {
      kiss_fftr_free(frontend->fft);
    }
    free(frontend);
  }
}

int whisper_frontend_compute(whisper_frontend *frontend,
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
                             size_t error_size) {
  if (frontend == NULL || frontend->fft == NULL) {
    return fail(error, error_size, "frontend FFT plan is unavailable");
  }
  if (hann_bytes == NULL || hann_size != WHISPER_FRONTEND_HANN_BYTES) {
    return fail(error, error_size, "Hann table must contain 400 float32 values");
  }
  if (mel_bytes == NULL || mel_size != WHISPER_FRONTEND_MEL_BYTES) {
    return fail(error, error_size, "mel table must have shape [201,80] float32");
  }
  if (output_u16 == NULL || output_elements != WHISPER_FRONTEND_ELEMENTS) {
    return fail(error, error_size, "output must have shape [1,80,3000] uint16");
  }

  pcm_view pcm;
  if (!parse_pcm16_wav(wav_bytes, wav_size, expected_pcm_frames, &pcm, error,
                       error_size)) {
    return 0;
  }
  if (fesetround(FE_TONEAREST) != 0) {
    return fail(error, error_size, "cannot select ties-to-even rounding");
  }

  float *samples = (float *)calloc(WHISPER_FRONTEND_PADDED_SAMPLES, sizeof(float));
  float *features = (float *)malloc(WHISPER_FRONTEND_ELEMENTS * sizeof(float));
  float *hann = (float *)malloc(WHISPER_FRONTEND_FFT_SIZE * sizeof(float));
  float *mel = (float *)malloc(WHISPER_FRONTEND_BINS * WHISPER_FRONTEND_MELS *
                               sizeof(float));
  if (samples == NULL || features == NULL || hann == NULL || mel == NULL) {
    free(samples);
    free(features);
    free(hann);
    free(mel);
    return fail(error, error_size, "cannot allocate frontend work buffers");
  }

  for (uint32_t index = 0; index < pcm.frames; ++index) {
    const uint16_t raw = read_u16_le(pcm.samples + index * 2U);
    const int32_t encoded = raw >= 0x8000U ? (int32_t)raw - 65536
                                          : (int32_t)raw;
    samples[index] = (float)encoded / 32768.0f;
  }
  for (int index = 0; index < WHISPER_FRONTEND_FFT_SIZE; ++index) {
    hann[index] = read_f32_le(hann_bytes + (size_t)index * 4U);
  }
  for (int index = 0; index < WHISPER_FRONTEND_BINS * WHISPER_FRONTEND_MELS;
       ++index) {
    mel[index] = read_f32_le(mel_bytes + (size_t)index * 4U);
  }

  float frame[WHISPER_FRONTEND_FFT_SIZE];
  kiss_fft_cpx spectrum[WHISPER_FRONTEND_BINS];
  float power[WHISPER_FRONTEND_BINS];
  for (int frame_index = 0; frame_index < WHISPER_FRONTEND_FRAMES;
       ++frame_index) {
    const int64_t start =
        (int64_t)frame_index * WHISPER_FRONTEND_HOP_SIZE -
        WHISPER_FRONTEND_FFT_SIZE / 2;
    for (int sample_index = 0; sample_index < WHISPER_FRONTEND_FFT_SIZE;
         ++sample_index) {
      const int32_t reflected = torch_reflect_index(
          start + sample_index, WHISPER_FRONTEND_PADDED_SAMPLES);
      frame[sample_index] = samples[reflected] * hann[sample_index];
    }
    kiss_fftr(frontend->fft, frame, spectrum);
    for (int bin = 0; bin < WHISPER_FRONTEND_BINS; ++bin) {
      const float magnitude = hypotf(spectrum[bin].r, spectrum[bin].i);
      power[bin] = magnitude * magnitude;
    }
    for (int mel_index = 0; mel_index < WHISPER_FRONTEND_MELS; ++mel_index) {
      float energy = 0.0f;
      for (int bin = 0; bin < WHISPER_FRONTEND_BINS; ++bin) {
        const float weight =
            mel[bin * WHISPER_FRONTEND_MELS + mel_index];
        energy = fmaf(weight, power[bin], energy);
      }
      features[mel_index * WHISPER_FRONTEND_FRAMES + frame_index] = energy;
    }
  }

  float maximum = -INFINITY;
  for (size_t index = 0; index < WHISPER_FRONTEND_ELEMENTS; ++index) {
    const float floored = fmaxf(features[index], 1.0e-10f);
    const float value = log10f(floored);
    features[index] = value;
    maximum = fmaxf(maximum, value);
  }
  const float lower_bound = maximum - 8.0f;
  for (size_t index = 0; index < WHISPER_FRONTEND_ELEMENTS; ++index) {
    const float normalized =
        (fmaxf(features[index], lower_bound) + 4.0f) / 4.0f;
    double quantized = nearbyint((double)normalized /
                                 kFeatureQuantizationScale) +
                       kFeatureZeroPoint;
    quantized = fmax(0.0, fmin(65535.0, quantized));
    output_u16[index] = (uint16_t)quantized;
  }

  free(samples);
  free(features);
  free(hann);
  free(mel);
  if (error != NULL && error_size > 0) {
    error[0] = '\0';
  }
  return 1;
}
