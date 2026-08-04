// Lightweight host comparator for the QAI/Hugging Face Whisper frontend.
//
// This is deliberately a standalone audit binary, not Android application
// code.  The Python driver pins all inputs and compares every emitted stage
// against the committed frontend golden.

#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/whisper-feature.h"
#include "kiss_fftr.h"

namespace {

constexpr int32_t kSamples = 480000;
constexpr int32_t kFft = 400;
constexpr int32_t kHop = 160;
constexpr int32_t kBins = kFft / 2 + 1;
constexpr int32_t kMels = 80;
constexpr int32_t kFrames = 3000;
constexpr double kQuantizationScale = 4.677007018472068e-05;
constexpr int32_t kZeroPoint = 32072;

template <typename T>
std::vector<T> ReadExact(const std::string &path, size_t count) {
  static_assert(std::is_trivially_copyable<T>::value,
                "binary input type must be trivially copyable");
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open input: " + path);
  }
  const std::streamsize expected =
      static_cast<std::streamsize>(count * sizeof(T));
  if (stream.tellg() != expected) {
    throw std::runtime_error("unexpected byte count for input: " + path);
  }
  stream.seekg(0);
  std::vector<T> value(count);
  if (!stream.read(reinterpret_cast<char *>(value.data()), expected)) {
    throw std::runtime_error("cannot read input: " + path);
  }
  return value;
}

template <typename T>
void WriteExact(const std::string &path, const std::vector<T> &value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "binary output type must be trivially copyable");
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("cannot open output: " + path);
  }
  const std::streamsize bytes =
      static_cast<std::streamsize>(value.size() * sizeof(T));
  if (!stream.write(reinterpret_cast<const char *>(value.data()), bytes)) {
    throw std::runtime_error("cannot write output: " + path);
  }
}

int32_t TorchReflectIndex(int64_t index, int32_t size) {
  while (index < 0 || index >= size) {
    if (index < 0) {
      // torch.nn.functional.pad(..., mode="reflect"): -1 -> 1.
      index = -index;
    } else {
      // The first element beyond the right edge maps to size - 2.
      index = 2LL * size - 2 - index;
    }
  }
  return static_cast<int32_t>(index);
}

enum class PowerMode { kDirect, kHypot };
enum class MelMode { kScalar, kFma };

struct ExactResult {
  std::vector<float> frame0;
  std::vector<float> power;
  std::vector<float> mel_energies;
};

ExactResult ComputeExactKissFft(const std::vector<float> &samples,
                               const std::vector<float> &hann,
                               const std::vector<float> &mel_filters,
                               PowerMode power_mode, MelMode mel_mode) {
  kiss_fftr_cfg config = kiss_fftr_alloc(kFft, 0, nullptr, nullptr);
  if (config == nullptr) {
    throw std::runtime_error("kiss_fftr_alloc failed");
  }

  ExactResult result;
  result.frame0.resize(kFft);
  result.power.resize(static_cast<size_t>(kBins) * kFrames);
  result.mel_energies.resize(static_cast<size_t>(kMels) * kFrames);

  std::array<float, kFft> frame{};
  std::array<kiss_fft_cpx, kBins> spectrum{};
  std::array<float, kBins> frame_power{};

  for (int32_t frame_index = 0; frame_index < kFrames; ++frame_index) {
    const int64_t start = static_cast<int64_t>(frame_index) * kHop - kFft / 2;
    for (int32_t sample_index = 0; sample_index < kFft; ++sample_index) {
      const int32_t reflected =
          TorchReflectIndex(start + sample_index, kSamples);
      frame[sample_index] = samples[reflected] * hann[sample_index];
    }
    if (frame_index == 0) {
      std::copy(frame.begin(), frame.end(), result.frame0.begin());
    }

    kiss_fftr(config, frame.data(), spectrum.data());
    for (int32_t bin = 0; bin < kBins; ++bin) {
      const float real = spectrum[bin].r;
      const float imag = spectrum[bin].i;
      float value;
      if (power_mode == PowerMode::kHypot) {
        // The QAI/HF golden spells this as complex.abs() ** 2, which is not
        // bit-equivalent to real*real + imag*imag in PyTorch float32.
        const float magnitude = std::hypot(real, imag);
        value = magnitude * magnitude;
      } else {
        value = real * real + imag * imag;
      }
      frame_power[bin] = value;
      result.power[static_cast<size_t>(bin) * kFrames + frame_index] = value;
    }

    for (int32_t mel = 0; mel < kMels; ++mel) {
      float energy = 0.0f;
      for (int32_t bin = 0; bin < kBins; ++bin) {
        const float weight = mel_filters[static_cast<size_t>(bin) * kMels + mel];
        if (mel_mode == MelMode::kFma) {
          energy = std::fma(weight, frame_power[bin], energy);
        } else {
          energy += weight * frame_power[bin];
        }
      }
      result.mel_energies[static_cast<size_t>(mel) * kFrames + frame_index] =
          energy;
    }
  }

  kiss_fftr_free(config);
  return result;
}

struct StockResult {
  std::vector<float> frame0;
  std::vector<float> mel_energies;
  std::vector<float> hann;
  std::vector<float> mel_filters;
};

StockResult ComputeStockKaldi(const std::vector<float> &samples) {
  knf::WhisperFeatureOptions options;
  options.dim = kMels;
  knf::WhisperFeatureComputer computer(options);
  const knf::FrameExtractionOptions &frame_options = computer.GetFrameOptions();
  knf::FeatureWindowFunction window_function(frame_options);

  const int32_t frames = knf::NumFrames(samples.size(), frame_options, true);
  if (frames != kFrames) {
    throw std::runtime_error("stock kaldi frontend emitted unexpected frame count");
  }

  StockResult result;
  result.frame0.resize(kFft);
  result.mel_energies.resize(static_cast<size_t>(kMels) * kFrames);
  std::vector<float> frame;
  std::array<float, kMels> energies{};
  for (int32_t frame_index = 0; frame_index < kFrames; ++frame_index) {
    float unused_log_energy = 0.0f;
    knf::ExtractWindow(0, samples, frame_index, frame_options, window_function,
                       &frame, &unused_log_energy);
    if (frame_index == 0) {
      std::copy(frame.begin(), frame.end(), result.frame0.begin());
    }
    computer.Compute(0.0f, 1.0f, &frame, energies.data());
    for (int32_t mel = 0; mel < kMels; ++mel) {
      result.mel_energies[static_cast<size_t>(mel) * kFrames + frame_index] =
          energies[mel];
    }
  }

  result.hann = knf::GetWindow("hann", kFft);
  knf::MelBanksOptions mel_options;
  mel_options.num_bins = kMels;
  mel_options.low_freq = 0;
  mel_options.is_librosa = true;
  knf::MelBanks mel_banks(mel_options, frame_options, 1.0f);
  result.mel_filters = mel_banks.GetMatrix();  // [80, 201], row-major.
  return result;
}

struct NormalizedResult {
  std::vector<float> features;
  std::vector<uint16_t> quantized;
};

NormalizedResult NormalizeAndQuantize(const std::vector<float> &mel_energies) {
  NormalizedResult result;
  result.features.resize(mel_energies.size());
  result.quantized.resize(mel_energies.size());

  float maximum = -std::numeric_limits<float>::infinity();
  for (size_t index = 0; index < mel_energies.size(); ++index) {
    const float floored = std::max(mel_energies[index], 1.0e-10f);
    const float value = std::log10(floored);
    result.features[index] = value;
    maximum = std::max(maximum, value);
  }
  const float lower_bound = maximum - 8.0f;
  for (size_t index = 0; index < result.features.size(); ++index) {
    const float value =
        (std::max(result.features[index], lower_bound) + 4.0f) / 4.0f;
    result.features[index] = value;
    double rounded = std::nearbyint(static_cast<double>(value) /
                                    kQuantizationScale) +
                     kZeroPoint;
    rounded = std::max(0.0, std::min(65535.0, rounded));
    result.quantized[index] = static_cast<uint16_t>(rounded);
  }
  return result;
}

void Usage(const char *program) {
  std::cerr << "usage: " << program
            << " MODE PCM_F32 HANN_F32 MEL_F32 OUTPUT_PREFIX\n"
               "MODE is kissfft-direct-scalar, kissfft-hypot-scalar, "
               "kissfft-hypot-fma, or kaldi-stock\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 6) {
      Usage(argv[0]);
      return 2;
    }
    if (std::fesetround(FE_TONEAREST) != 0) {
      throw std::runtime_error("cannot select ties-to-even rounding mode");
    }
    const std::string mode = argv[1];
    const std::string prefix = argv[5];
    const std::vector<float> samples = ReadExact<float>(argv[2], kSamples);
    const auto begin = std::chrono::steady_clock::now();

    std::vector<float> mel_energies;
    if (mode == "kaldi-stock") {
      StockResult stock = ComputeStockKaldi(samples);
      WriteExact(prefix + ".frame0.f32le.bin", stock.frame0);
      WriteExact(prefix + ".hann.f32le.bin", stock.hann);
      WriteExact(prefix + ".mel_filters.f32le.bin", stock.mel_filters);
      mel_energies = std::move(stock.mel_energies);
    } else {
      const std::vector<float> hann = ReadExact<float>(argv[3], kFft);
      const std::vector<float> mel_filters =
          ReadExact<float>(argv[4], static_cast<size_t>(kBins) * kMels);
      PowerMode power_mode;
      MelMode mel_mode;
      if (mode == "kissfft-direct-scalar") {
        power_mode = PowerMode::kDirect;
        mel_mode = MelMode::kScalar;
      } else if (mode == "kissfft-hypot-scalar") {
        power_mode = PowerMode::kHypot;
        mel_mode = MelMode::kScalar;
      } else if (mode == "kissfft-hypot-fma") {
        power_mode = PowerMode::kHypot;
        mel_mode = MelMode::kFma;
      } else {
        Usage(argv[0]);
        return 2;
      }
      ExactResult exact = ComputeExactKissFft(samples, hann, mel_filters,
                                              power_mode, mel_mode);
      WriteExact(prefix + ".frame0.f32le.bin", exact.frame0);
      WriteExact(prefix + ".power.f32le.bin", exact.power);
      mel_energies = std::move(exact.mel_energies);
    }

    WriteExact(prefix + ".mel_energies.f32le.bin", mel_energies);
    const NormalizedResult normalized = NormalizeAndQuantize(mel_energies);
    WriteExact(prefix + ".features.f32le.bin", normalized.features);
    WriteExact(prefix + ".features.u16le.bin", normalized.quantized);

    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    std::cout << "{\"mode\":\"" << mode << "\",\"seconds\":" << seconds
              << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "native frontend audit failed: " << error.what() << "\n";
    return 1;
  }
}
