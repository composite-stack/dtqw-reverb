#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace dtqw {

enum class WavBitDepth {
    Bit16,
    Bit24,
    Float32
};

struct WavStats {
    float peak_dbfs_l = -120.0f;
    float peak_dbfs_r = -120.0f;
    float peak_linear_l = 0.0f;
    float peak_linear_r = 0.0f;
    double sum_sq_l = 0.0;
    double sum_sq_r = 0.0;
    size_t total_frames = 0;

    void update(const float* l, const float* r, size_t num_frames) noexcept {
        for (size_t i = 0; i < num_frames; ++i) {
            float abs_l = std::abs(l[i]);
            float abs_r = std::abs(r[i]);
            if (abs_l > peak_linear_l) peak_linear_l = abs_l;
            if (abs_r > peak_linear_r) peak_linear_r = abs_r;
            sum_sq_l += l[i] * l[i];
            sum_sq_r += r[i] * r[i];
        }
        total_frames += num_frames;
    }

    void finalize() noexcept {
        peak_dbfs_l = (peak_linear_l > 1e-6f) ? 20.0f * std::log10(peak_linear_l) : -120.0f;
        peak_dbfs_r = (peak_linear_r > 1e-6f) ? 20.0f * std::log10(peak_linear_r) : -120.0f;
    }

    float getRmsDbfsL() const noexcept {
        if (total_frames == 0) return -120.0f;
        double rms = std::sqrt(sum_sq_l / total_frames);
        return (rms > 1e-6) ? static_cast<float>(20.0 * std::log10(rms)) : -120.0f;
    }

    float getRmsDbfsR() const noexcept {
        if (total_frames == 0) return -120.0f;
        double rms = std::sqrt(sum_sq_r / total_frames);
        return (rms > 1e-6) ? static_cast<float>(20.0 * std::log10(rms)) : -120.0f;
    }

    float getMaxPeakDbfs() const noexcept {
        return std::max(peak_dbfs_l, peak_dbfs_r);
    }

    float getMaxRmsDbfs() const noexcept {
        return std::max(getRmsDbfsL(), getRmsDbfsR());
    }

    float getCrestFactorDb() const noexcept {
        return getMaxPeakDbfs() - getMaxRmsDbfs();
    }
};

class WavReader {
public:
    WavReader() = default;
    ~WavReader() { close(); }

    bool open(const std::string& path) {
        close();
        file_.open(path, std::ios::binary);
        if (!file_.is_open()) {
            error_ = "Failed to open file: " + path;
            return false;
        }

        char riff[4];
        uint32_t riff_size;
        char wave[4];

        if (!file_.read(riff, 4) || !file_.read(reinterpret_cast<char*>(&riff_size), 4) || !file_.read(wave, 4)) {
            error_ = "Failed to read RIFF header";
            return false;
        }

        if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
            error_ = "Not a valid RIFF/WAVE file";
            return false;
        }

        bool found_fmt = false;
        bool found_data = false;

        while (file_) {
            char chunk_id[4];
            uint32_t chunk_size = 0;
            if (!file_.read(chunk_id, 4) || !file_.read(reinterpret_cast<char*>(&chunk_size), 4)) {
                break;
            }

            if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
                if (chunk_size < 16) {
                    error_ = "Corrupt fmt chunk";
                    return false;
                }
                uint16_t format_tag = 0;
                file_.read(reinterpret_cast<char*>(&format_tag), 2);
                file_.read(reinterpret_cast<char*>(&channels_), 2);
                file_.read(reinterpret_cast<char*>(&sample_rate_), 4);
                uint32_t byte_rate = 0;
                uint16_t block_align = 0;
                file_.read(reinterpret_cast<char*>(&byte_rate), 4);
                file_.read(reinterpret_cast<char*>(&block_align), 2);
                file_.read(reinterpret_cast<char*>(&bits_per_sample_), 2);

                if (format_tag == 0xFFFE) { // WAVE_FORMAT_EXTENSIBLE
                    if (chunk_size >= 40) {
                        uint16_t cb_size = 0;
                        file_.read(reinterpret_cast<char*>(&cb_size), 2);
                        uint16_t valid_bits = 0;
                        file_.read(reinterpret_cast<char*>(&valid_bits), 2);
                        uint32_t channel_mask = 0;
                        file_.read(reinterpret_cast<char*>(&channel_mask), 4);
                        uint16_t sub_format = 0;
                        file_.read(reinterpret_cast<char*>(&sub_format), 2);
                        format_tag = sub_format;
                        file_.seekg(chunk_size - 26, std::ios::cur);
                    } else {
                        file_.seekg(chunk_size - 16, std::ios::cur);
                    }
                } else if (chunk_size > 16) {
                    file_.seekg(chunk_size - 16, std::ios::cur);
                }

                format_tag_ = format_tag;
                found_fmt = true;
                if (found_data) break;
            } else if (std::memcmp(chunk_id, "data", 4) == 0) {
                data_size_ = chunk_size;
                data_offset_ = static_cast<std::streamoff>(file_.tellg());
                found_data = true;
                if (found_fmt) break;
                file_.seekg(chunk_size, std::ios::cur);
            } else {
                file_.seekg(chunk_size, std::ios::cur);
            }

            if (chunk_size % 2 != 0) {
                file_.seekg(1, std::ios::cur);
            }
        }

        if (!found_fmt || !found_data) {
            error_ = "Missing fmt or data chunk";
            return false;
        }

        if (channels_ == 0 || sample_rate_ == 0) {
            error_ = "Invalid channel count or sample rate";
            return false;
        }

        const bool supported_pcm = format_tag_ == 1 &&
            (bits_per_sample_ == 16 || bits_per_sample_ == 24 || bits_per_sample_ == 32);
        const bool supported_float = format_tag_ == 3 && bits_per_sample_ == 32;
        if (!supported_pcm && !supported_float) {
            error_ = "Unsupported WAV encoding (expected PCM 16/24/32-bit or IEEE float 32-bit)";
            return false;
        }
        size_t bytes_per_sample = bits_per_sample_ / 8;

        bytes_per_frame_ = channels_ * bytes_per_sample;
        total_frames_ = data_size_ / bytes_per_frame_;
        frames_remaining_ = total_frames_;

        file_.seekg(data_offset_, std::ios::beg);
        return true;
    }

    void close() {
        if (file_.is_open()) {
            file_.close();
        }
        channels_ = 0;
        sample_rate_ = 0;
        bits_per_sample_ = 0;
        format_tag_ = 0;
        total_frames_ = 0;
        frames_remaining_ = 0;
        data_offset_ = 0;
        data_size_ = 0;
    }

    size_t readFrames(float* out_l, float* out_r, size_t frames_to_read) {
        if (!file_.is_open() || frames_remaining_ == 0 || frames_to_read == 0) {
            return 0;
        }

        size_t actual_frames = std::min(frames_to_read, frames_remaining_);
        size_t total_bytes = actual_frames * bytes_per_frame_;
        raw_buffer_.resize(total_bytes);

        if (!file_.read(reinterpret_cast<char*>(raw_buffer_.data()), total_bytes)) {
            actual_frames = file_.gcount() / bytes_per_frame_;
        }

        frames_remaining_ -= actual_frames;
        const uint8_t* ptr = raw_buffer_.data();

        if (format_tag_ == 3) { // 32-bit IEEE float
            for (size_t i = 0; i < actual_frames; ++i) {
                float val_l = 0.0f;
                std::memcpy(&val_l, ptr, sizeof(float));
                ptr += 4;
                float val_r = val_l;
                if (channels_ > 1) {
                    std::memcpy(&val_r, ptr, sizeof(float));
                    ptr += 4 * (channels_ - 1);
                }
                out_l[i] = val_l;
                out_r[i] = val_r;
            }
        } else if (bits_per_sample_ == 16) { // 16-bit PCM
            for (size_t i = 0; i < actual_frames; ++i) {
                int16_t val_l = static_cast<int16_t>(ptr[0] | (ptr[1] << 8));
                ptr += 2;
                int16_t val_r = val_l;
                if (channels_ > 1) {
                    val_r = static_cast<int16_t>(ptr[0] | (ptr[1] << 8));
                    ptr += 2 * (channels_ - 1);
                }
                out_l[i] = val_l * (1.0f / 32768.0f);
                out_r[i] = val_r * (1.0f / 32768.0f);
            }
        } else if (bits_per_sample_ == 24) { // 24-bit PCM
            for (size_t i = 0; i < actual_frames; ++i) {
                int32_t s_l = static_cast<int32_t>(ptr[0]) |
                             (static_cast<int32_t>(ptr[1]) << 8) |
                             (static_cast<int32_t>(ptr[2]) << 16);
                if (s_l & 0x00800000) s_l |= 0xFF000000;
                ptr += 3;

                int32_t s_r = s_l;
                if (channels_ > 1) {
                    s_r = static_cast<int32_t>(ptr[0]) |
                          (static_cast<int32_t>(ptr[1]) << 8) |
                          (static_cast<int32_t>(ptr[2]) << 16);
                    if (s_r & 0x00800000) s_r |= 0xFF000000;
                    ptr += 3 * (channels_ - 1);
                }
                out_l[i] = s_l * (1.0f / 8388608.0f);
                out_r[i] = s_r * (1.0f / 8388608.0f);
            }
        } else if (bits_per_sample_ == 32) { // 32-bit Integer PCM
            for (size_t i = 0; i < actual_frames; ++i) {
                int32_t s_l = 0;
                std::memcpy(&s_l, ptr, sizeof(int32_t));
                ptr += 4;
                int32_t s_r = s_l;
                if (channels_ > 1) {
                    std::memcpy(&s_r, ptr, sizeof(int32_t));
                    ptr += 4 * (channels_ - 1);
                }
                out_l[i] = s_l * (1.0f / 2147483648.0f);
                out_r[i] = s_r * (1.0f / 2147483648.0f);
            }
        } else {
            for (size_t i = 0; i < actual_frames; ++i) {
                out_l[i] = 0.0f;
                out_r[i] = 0.0f;
            }
        }

        return actual_frames;
    }

    uint32_t getSampleRate() const noexcept { return sample_rate_; }
    uint16_t getChannels() const noexcept { return channels_; }
    uint16_t getBitsPerSample() const noexcept { return bits_per_sample_; }
    uint16_t getFormatTag() const noexcept { return format_tag_; }
    size_t getTotalFrames() const noexcept { return total_frames_; }
    size_t getFramesRemaining() const noexcept { return frames_remaining_; }
    double getDurationSeconds() const noexcept {
        return (sample_rate_ > 0) ? static_cast<double>(total_frames_) / sample_rate_ : 0.0;
    }
    const std::string& getError() const noexcept { return error_; }

private:
    std::ifstream file_;
    std::string error_;
    uint32_t sample_rate_{0};
    uint16_t channels_{0};
    uint16_t bits_per_sample_{0};
    uint16_t format_tag_{0};
    size_t total_frames_{0};
    size_t frames_remaining_{0};
    size_t bytes_per_frame_{0};
    std::streamoff data_offset_{0};
    uint32_t data_size_{0};
    std::vector<uint8_t> raw_buffer_;
};

class WavWriter {
public:
    WavWriter() = default;
    ~WavWriter() { close(); }

    bool open(const std::string& path, uint32_t sample_rate, uint16_t channels, WavBitDepth bit_depth) {
        close();
        file_.open(path, std::ios::binary);
        if (!file_.is_open()) {
            error_ = "Cannot create output file: " + path;
            return false;
        }

        sample_rate_ = sample_rate;
        channels_ = channels;
        bit_depth_ = bit_depth;
        frames_written_ = 0;

        writeHeaderPlaceholder();
        return true;
    }

    void writeFrames(const float* in_l, const float* in_r, size_t num_frames) {
        if (!file_.is_open() || num_frames == 0) return;

        if (bit_depth_ == WavBitDepth::Float32) {
            raw_buffer_.resize(num_frames * channels_ * 4);
            uint8_t* dst = raw_buffer_.data();
            for (size_t i = 0; i < num_frames; ++i) {
                std::memcpy(dst, &in_l[i], sizeof(float));
                dst += sizeof(float);
                if (channels_ > 1) {
                    std::memcpy(dst, &in_r[i], sizeof(float));
                    dst += sizeof(float);
                }
            }
            file_.write(reinterpret_cast<const char*>(raw_buffer_.data()), raw_buffer_.size());
        } else if (bit_depth_ == WavBitDepth::Bit24) {
            raw_buffer_.resize(num_frames * channels_ * 3);
            uint8_t* dst = raw_buffer_.data();
            for (size_t i = 0; i < num_frames; ++i) {
                float cl = std::max(-1.0f, std::min(in_l[i], 1.0f));
                int32_t vl = static_cast<int32_t>(std::round(cl * 8388607.0f));
                *dst++ = static_cast<uint8_t>(vl & 0xFF);
                *dst++ = static_cast<uint8_t>((vl >> 8) & 0xFF);
                *dst++ = static_cast<uint8_t>((vl >> 16) & 0xFF);

                if (channels_ > 1) {
                    float cr = std::max(-1.0f, std::min(in_r[i], 1.0f));
                    int32_t vr = static_cast<int32_t>(std::round(cr * 8388607.0f));
                    *dst++ = static_cast<uint8_t>(vr & 0xFF);
                    *dst++ = static_cast<uint8_t>((vr >> 8) & 0xFF);
                    *dst++ = static_cast<uint8_t>((vr >> 16) & 0xFF);
                }
            }
            file_.write(reinterpret_cast<const char*>(raw_buffer_.data()), raw_buffer_.size());
        } else if (bit_depth_ == WavBitDepth::Bit16) {
            raw_buffer_.resize(num_frames * channels_ * 2);
            uint8_t* dst = raw_buffer_.data();
            for (size_t i = 0; i < num_frames; ++i) {
                float cl = std::max(-1.0f, std::min(in_l[i], 1.0f));
                int16_t vl = static_cast<int16_t>(std::round(cl * 32767.0f));
                std::memcpy(dst, &vl, sizeof(int16_t));
                dst += sizeof(int16_t);
                if (channels_ > 1) {
                    float cr = std::max(-1.0f, std::min(in_r[i], 1.0f));
                    int16_t vr = static_cast<int16_t>(std::round(cr * 32767.0f));
                    std::memcpy(dst, &vr, sizeof(int16_t));
                    dst += sizeof(int16_t);
                }
            }
            file_.write(reinterpret_cast<const char*>(raw_buffer_.data()), raw_buffer_.size());
        }

        frames_written_ += num_frames;
    }

    void close() {
        if (!file_.is_open()) return;
        finalizeHeader();
        file_.close();
    }

    size_t getFramesWritten() const noexcept { return frames_written_; }
    const std::string& getError() const noexcept { return error_; }

private:
    void writeHeaderPlaceholder() {
        char riff[4] = {'R', 'I', 'F', 'F'};
        uint32_t dummy_size = 0;
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt[4] = {'f', 'm', 't', ' '};
        uint32_t fmt_size = 16;
        uint16_t format_tag = (bit_depth_ == WavBitDepth::Float32) ? 3 : 1;
        uint16_t channels = channels_;
        uint32_t sample_rate = sample_rate_;
        uint16_t bits_per_sample = (bit_depth_ == WavBitDepth::Float32) ? 32 : ((bit_depth_ == WavBitDepth::Bit24) ? 24 : 16);
        uint16_t block_align = channels * (bits_per_sample / 8);
        uint32_t byte_rate = sample_rate * block_align;
        char data[4] = {'d', 'a', 't', 'a'};

        file_.write(riff, 4);
        file_.write(reinterpret_cast<char*>(&dummy_size), 4);
        file_.write(wave, 4);
        file_.write(fmt, 4);
        file_.write(reinterpret_cast<char*>(&fmt_size), 4);
        file_.write(reinterpret_cast<char*>(&format_tag), 2);
        file_.write(reinterpret_cast<char*>(&channels), 2);
        file_.write(reinterpret_cast<char*>(&sample_rate), 4);
        file_.write(reinterpret_cast<char*>(&byte_rate), 4);
        file_.write(reinterpret_cast<char*>(&block_align), 2);
        file_.write(reinterpret_cast<char*>(&bits_per_sample), 2);
        file_.write(data, 4);
        file_.write(reinterpret_cast<char*>(&dummy_size), 4);
    }

    void finalizeHeader() {
        uint16_t bits_per_sample = (bit_depth_ == WavBitDepth::Float32) ? 32 : ((bit_depth_ == WavBitDepth::Bit24) ? 24 : 16);
        uint32_t bytes_per_frame = channels_ * (bits_per_sample / 8);
        uint32_t data_bytes = static_cast<uint32_t>(frames_written_ * bytes_per_frame);
        uint32_t riff_size = 36 + data_bytes;

        file_.seekp(4, std::ios::beg);
        file_.write(reinterpret_cast<char*>(&riff_size), 4);

        file_.seekp(40, std::ios::beg);
        file_.write(reinterpret_cast<char*>(&data_bytes), 4);
    }

    std::ofstream file_;
    std::string error_;
    uint32_t sample_rate_{48000};
    uint16_t channels_{2};
    WavBitDepth bit_depth_{WavBitDepth::Bit24};
    size_t frames_written_{0};
    std::vector<uint8_t> raw_buffer_;
};

} // namespace dtqw
