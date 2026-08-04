#include <jni.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

#include "whisper_frontend_core.h"

static pthread_once_t g_frontend_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_frontend_mutex = PTHREAD_MUTEX_INITIALIZER;
static whisper_frontend *g_frontend = NULL;
static char g_frontend_error[256];

static void initialize_frontend(void) {
  g_frontend = whisper_frontend_create(g_frontend_error,
                                       sizeof(g_frontend_error));
}

static void throw_exception(JNIEnv *environment,
                            const char *class_name,
                            const char *message) {
  jclass exception_class = (*environment)->FindClass(environment, class_name);
  if (exception_class != NULL) {
    (*environment)->ThrowNew(environment, exception_class, message);
  }
}

static uint8_t *copy_byte_array(JNIEnv *environment,
                                jbyteArray source,
                                jsize *length,
                                const char *label) {
  if (source == NULL) {
    throw_exception(environment, "java/lang/IllegalArgumentException", label);
    return NULL;
  }
  *length = (*environment)->GetArrayLength(environment, source);
  uint8_t *copy = (uint8_t *)malloc((size_t)*length);
  if (copy == NULL) {
    throw_exception(environment, "java/lang/OutOfMemoryError", label);
    return NULL;
  }
  (*environment)->GetByteArrayRegion(environment, source, 0, *length,
                                     (jbyte *)copy);
  if ((*environment)->ExceptionCheck(environment)) {
    free(copy);
    return NULL;
  }
  return copy;
}

JNIEXPORT jbyteArray JNICALL
Java_ai_liquid_lfmdemo_whisperprobe_WhisperFrontendNative_compute(
    JNIEnv *environment,
    jclass unused_class,
    jbyteArray wav_array,
    jint expected_pcm_frames,
    jbyteArray hann_array,
    jbyteArray mel_array) {
  (void)unused_class;
  if (expected_pcm_frames < 0) {
    throw_exception(environment, "java/lang/IllegalArgumentException",
                    "expected PCM frames must be nonnegative");
    return NULL;
  }

  jsize wav_size = 0;
  jsize hann_size = 0;
  jsize mel_size = 0;
  uint8_t *wav = copy_byte_array(environment, wav_array, &wav_size,
                                 "WAV byte array is required");
  if (wav == NULL) {
    return NULL;
  }
  uint8_t *hann = copy_byte_array(environment, hann_array, &hann_size,
                                  "Hann byte array is required");
  if (hann == NULL) {
    free(wav);
    return NULL;
  }
  uint8_t *mel = copy_byte_array(environment, mel_array, &mel_size,
                                 "mel byte array is required");
  if (mel == NULL) {
    free(wav);
    free(hann);
    return NULL;
  }
  uint16_t *output = (uint16_t *)malloc(WHISPER_FRONTEND_OUTPUT_BYTES);
  if (output == NULL) {
    free(wav);
    free(hann);
    free(mel);
    throw_exception(environment, "java/lang/OutOfMemoryError",
                    "cannot allocate Whisper frontend output");
    return NULL;
  }

  (void)pthread_once(&g_frontend_once, initialize_frontend);
  if (g_frontend == NULL) {
    free(wav);
    free(hann);
    free(mel);
    free(output);
    throw_exception(environment, "java/lang/IllegalStateException",
                    g_frontend_error[0] == '\0' ? "frontend initialization failed"
                                                : g_frontend_error);
    return NULL;
  }

  char error[256];
  (void)pthread_mutex_lock(&g_frontend_mutex);
  const int success = whisper_frontend_compute(
      g_frontend, wav, (size_t)wav_size, (uint32_t)expected_pcm_frames, hann,
      (size_t)hann_size, mel, (size_t)mel_size, output,
      WHISPER_FRONTEND_ELEMENTS, error, sizeof(error));
  (void)pthread_mutex_unlock(&g_frontend_mutex);
  free(wav);
  free(hann);
  free(mel);
  if (!success) {
    free(output);
    throw_exception(environment, "java/lang/IllegalArgumentException", error);
    return NULL;
  }

  jbyteArray encoded = (*environment)->NewByteArray(
      environment, WHISPER_FRONTEND_OUTPUT_BYTES);
  if (encoded != NULL) {
    (*environment)->SetByteArrayRegion(
        environment, encoded, 0, WHISPER_FRONTEND_OUTPUT_BYTES,
        (const jbyte *)output);
  }
  free(output);
  return encoded;
}
