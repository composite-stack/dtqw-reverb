#include "WavFile.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void testRoundTrip(dtqw::WavBitDepth depth, const std::string& path, float tolerance) {
    constexpr uint32_t sample_rate = 48000;
    constexpr size_t frames = 257;
    std::vector<float> left(frames), right(frames), decoded_left(frames), decoded_right(frames);

    for (size_t i = 0; i < frames; ++i) {
        left[i] = 0.7f * std::sin(static_cast<float>(i) * 0.07f);
        right[i] = 0.6f * std::cos(static_cast<float>(i) * 0.11f);
    }

    dtqw::WavWriter writer;
    assert(writer.open(path, sample_rate, 2, depth));
    writer.writeFrames(left.data(), right.data(), frames);
    writer.close();

    dtqw::WavReader reader;
    assert(reader.open(path));
    assert(reader.getSampleRate() == sample_rate);
    assert(reader.getChannels() == 2);
    assert(reader.getTotalFrames() == frames);
    assert(reader.readFrames(decoded_left.data(), decoded_right.data(), frames) == frames);

    float max_error = 0.0f;
    for (size_t i = 0; i < frames; ++i) {
        max_error = std::max(max_error, std::abs(left[i] - decoded_left[i]));
        max_error = std::max(max_error, std::abs(right[i] - decoded_right[i]));
    }
    assert(max_error <= tolerance);

    reader.close();
    std::filesystem::remove(path);
}

void testMonoInputDuplicatesToStereo(const std::string& path) {
    constexpr uint32_t sample_rate = 48000;
    constexpr size_t frames = 32;
    std::vector<float> mono(frames), unused(frames), left(frames), right(frames);
    for (size_t i = 0; i < frames; ++i) mono[i] = static_cast<float>(i) / frames - 0.5f;

    dtqw::WavWriter writer;
    assert(writer.open(path, sample_rate, 1, dtqw::WavBitDepth::Float32));
    writer.writeFrames(mono.data(), unused.data(), frames);
    writer.close();

    dtqw::WavReader reader;
    assert(reader.open(path));
    assert(reader.getChannels() == 1);
    assert(reader.readFrames(left.data(), right.data(), frames) == frames);
    for (size_t i = 0; i < frames; ++i) assert(std::abs(left[i] - right[i]) < 1e-7f);
    reader.close();
    std::filesystem::remove(path);
}

void testStats() {
    const float left[] = {1.0f, -1.0f, 0.0f, 0.0f};
    const float right[] = {0.5f, -0.5f, 0.0f, 0.0f};
    dtqw::WavStats stats;
    stats.update(left, right, 4);
    stats.finalize();
    assert(std::abs(stats.peak_dbfs_l) < 1e-5f);
    assert(std::abs(stats.peak_dbfs_r + 6.0206f) < 0.01f);
    assert(std::abs(stats.getRmsDbfsL() + 3.0103f) < 0.01f);
}

} // namespace

int main() {
#ifdef NDEBUG
#error "WAV harness tests must be built with assertions enabled"
#endif
    std::cout << "DTQW WAV Harness Tests\n";
    testRoundTrip(dtqw::WavBitDepth::Bit16, "dtqw_test_16.wav", 4.0e-5f);
    testRoundTrip(dtqw::WavBitDepth::Bit24, "dtqw_test_24.wav", 2.0e-7f);
    testRoundTrip(dtqw::WavBitDepth::Float32, "dtqw_test_float.wav", 1.0e-7f);
    testMonoInputDuplicatesToStereo("dtqw_test_mono.wav");
    testStats();
    std::cout << "All WAV harness tests passed.\n";
    return 0;
}
