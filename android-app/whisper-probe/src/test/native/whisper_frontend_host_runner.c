#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "whisper_frontend_core.h"

static uint8_t *read_file(const char *path, size_t *size) {
  FILE *stream = fopen(path, "rb");
  if (stream == NULL) {
    fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
    return NULL;
  }
  if (fseek(stream, 0, SEEK_END) != 0) {
    fclose(stream);
    return NULL;
  }
  const long length = ftell(stream);
  if (length < 0 || fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    return NULL;
  }
  uint8_t *value = (uint8_t *)malloc((size_t)length);
  if (value == NULL || fread(value, 1, (size_t)length, stream) != (size_t)length) {
    free(value);
    fclose(stream);
    return NULL;
  }
  fclose(stream);
  *size = (size_t)length;
  return value;
}

static int write_u16_le(const char *path, const uint16_t *values, size_t count) {
  FILE *stream = fopen(path, "wb");
  if (stream == NULL) {
    return 0;
  }
  for (size_t index = 0; index < count; ++index) {
    const uint8_t encoded[2] = {
        (uint8_t)(values[index] & 0xffU),
        (uint8_t)(values[index] >> 8),
    };
    if (fwrite(encoded, 1, 2, stream) != 2) {
      fclose(stream);
      return 0;
    }
  }
  return fclose(stream) == 0;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr,
            "usage: %s WAV EXPECTED_PCM_FRAMES HANN MEL OUTPUT_U16LE\n",
            argv[0]);
    return 2;
  }
  char *end = NULL;
  const unsigned long frames = strtoul(argv[2], &end, 10);
  if (end == argv[2] || *end != '\0' || frames > UINT32_MAX) {
    fprintf(stderr, "invalid expected PCM frame count\n");
    return 2;
  }

  size_t wav_size = 0;
  size_t hann_size = 0;
  size_t mel_size = 0;
  uint8_t *wav = read_file(argv[1], &wav_size);
  uint8_t *hann = read_file(argv[3], &hann_size);
  uint8_t *mel = read_file(argv[4], &mel_size);
  uint16_t *output =
      (uint16_t *)malloc(WHISPER_FRONTEND_OUTPUT_BYTES);
  if (wav == NULL || hann == NULL || mel == NULL || output == NULL) {
    fprintf(stderr, "cannot load host frontend inputs\n");
    free(wav);
    free(hann);
    free(mel);
    free(output);
    return 1;
  }

  char error[256];
  whisper_frontend *frontend = whisper_frontend_create(error, sizeof(error));
  if (frontend == NULL) {
    fprintf(stderr, "%s\n", error);
    free(wav);
    free(hann);
    free(mel);
    free(output);
    return 1;
  }
  const int success = whisper_frontend_compute(
      frontend, wav, wav_size, (uint32_t)frames, hann, hann_size, mel, mel_size,
      output, WHISPER_FRONTEND_ELEMENTS, error, sizeof(error));
  whisper_frontend_destroy(frontend);
  free(wav);
  free(hann);
  free(mel);
  if (!success) {
    fprintf(stderr, "%s\n", error);
    free(output);
    return 1;
  }
  if (!write_u16_le(argv[5], output, WHISPER_FRONTEND_ELEMENTS)) {
    fprintf(stderr, "cannot write output\n");
    free(output);
    return 1;
  }
  free(output);
  return 0;
}
