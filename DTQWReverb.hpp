#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <utility>

#if defined(__SSE__) || defined(_M_X64)
  #include <xmmintrin.h>
  #include <emmintrin.h>
  #if defined(__FMA__)
    #include <immintrin.h>
  #endif
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
  #if defined(DTQW_TEST_NEON_ON_X86) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
    #include "NEON_2_SSE.h"
  #else
    #include <arm_neon.h>
  #endif
#elif defined(DTQW_TEST_NEON_ON_X86) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
  #define __ARM_NEON 1
  #define __aarch64__ 1
  #include "NEON_2_SSE.h"
#endif

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

namespace dtqw::simd {

#if defined(__SSE__) || defined(_M_X64)
struct SSEBackend {
    using Vec4 = __m128;

    static constexpr const char* name() noexcept { return "SSE4.1"; }

    static inline Vec4 setzero() noexcept { return _mm_setzero_ps(); }
    static inline Vec4 set1(float v) noexcept { return _mm_set1_ps(v); }
    static inline Vec4 loadu(const float* p) noexcept { return _mm_loadu_ps(p); }
    static inline void storeu(float* p, Vec4 v) noexcept { _mm_storeu_ps(p, v); }

    static inline Vec4 add(Vec4 a, Vec4 b) noexcept { return _mm_add_ps(a, b); }
    static inline Vec4 sub(Vec4 a, Vec4 b) noexcept { return _mm_sub_ps(a, b); }
    static inline Vec4 mul(Vec4 a, Vec4 b) noexcept { return _mm_mul_ps(a, b); }
    static inline Vec4 fma(Vec4 a, Vec4 b, Vec4 c) noexcept {
#if defined(__FMA__)
        return _mm_fmadd_ps(a, b, c);
#else
        return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
    }

    static inline Vec4 set(float x0, float x1, float x2, float x3) noexcept {
        return _mm_set_ps(x3, x2, x1, x0);
    }

    /// Parameterized unitary coin with continuous phase rotation theta:
    /// Eigenvalues rotate as exp(i*(pi -+ theta)) on the transversal scattering plane.
    /// theta = 0 corresponds to the standard Grover coin.
    static inline Vec4 coin(Vec4 v, Vec4 coeff_a, Vec4 coeff_odd, Vec4 coeff_even) noexcept {
        __m128 v_halves = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
        __m128 diff     = _mm_sub_ps(v, v_halves);
        __m128 term_a   = _mm_mul_ps(coeff_a, diff);

        __m128 v_odd    = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 0, 1));
        __m128 v_even   = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 2, 3));

#if defined(__FMA__)
        __m128 term_bd  = _mm_fmadd_ps(coeff_odd, v_odd, _mm_mul_ps(coeff_even, v_even));
#else
        __m128 term_bd  = _mm_add_ps(_mm_mul_ps(coeff_odd, v_odd), _mm_mul_ps(coeff_even, v_even));
#endif
        return _mm_add_ps(term_bd, term_a);
    }

    /// Fast 4-port Grover scattering matrix C = (1/2)*1*1^T - I
    static inline Vec4 grover(Vec4 v) noexcept {
        __m128 shuf1    = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sum2     = _mm_add_ps(v, shuf1);
        __m128 shuf2    = _mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(1, 0, 3, 2));
        __m128 sum4     = _mm_add_ps(sum2, shuf2);
        __m128 half_sum = _mm_mul_ps(sum4, _mm_set1_ps(0.5f));
        return _mm_sub_ps(half_sum, v);
    }

    static inline void transpose4x4(Vec4& r0, Vec4& r1, Vec4& r2, Vec4& r3) noexcept {
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    }

    /// 4-point, 3rd-order Hermite interpolation across 4 lanes
    static inline Vec4 hermite(Vec4 ym1, Vec4 y0, Vec4 y1, Vec4 y2, Vec4 frac) noexcept {
        const Vec4 half     = _mm_set1_ps(0.5f);
        const Vec4 one_half = _mm_set1_ps(1.5f);
        const Vec4 two      = _mm_set1_ps(2.0f);
        const Vec4 two_half = _mm_set1_ps(2.5f);

        Vec4 c0 = y0;
        Vec4 c1 = mul(half, sub(y1, ym1));
        Vec4 c2 = sub(add(ym1, mul(two, y1)), add(mul(two_half, y0), mul(half, y2)));
        Vec4 c3 = add(mul(half, sub(y2, ym1)), mul(one_half, sub(y0, y1)));

        Vec4 acc = fma(c3, frac, c2);
        acc = fma(acc, frac, c1);
        return fma(acc, frac, c0);
    }

    /// Branchless phase wrapping for phase in [0, 2) to [0, 1)
    static inline Vec4 wrap_01(Vec4 p) noexcept {
        const Vec4 one = _mm_set1_ps(1.0f);
        const Vec4 mask = _mm_cmpge_ps(p, one);
        return _mm_sub_ps(p, _mm_and_ps(mask, one));
    }

    /// Fast parabolic trigonometric approximation for normalized phase in [0, 1)
    /// Max error < 0.0011 (< 0.1% THD), branchless SSE
    static inline Vec4 fast_sin(Vec4 phase_01) noexcept {
        const Vec4 two   = _mm_set1_ps(2.0f);
        const Vec4 one   = _mm_set1_ps(1.0f);
        const Vec4 four  = _mm_set1_ps(4.0f);
        const Vec4 p0225 = _mm_set1_ps(0.225f);
        const Vec4 sign_mask = _mm_set1_ps(-0.0f);

        Vec4 x = _mm_sub_ps(_mm_mul_ps(two, phase_01), one); // [-1, 1)
        Vec4 abs_x = _mm_andnot_ps(sign_mask, x);
        Vec4 y = _mm_mul_ps(four, _mm_mul_ps(x, _mm_sub_ps(one, abs_x)));
        Vec4 abs_y = _mm_andnot_ps(sign_mask, y);
        Vec4 y_refine = _mm_add_ps(_mm_mul_ps(p0225, _mm_mul_ps(y, _mm_sub_ps(abs_y, one))), y);
        return _mm_sub_ps(_mm_setzero_ps(), y_refine);
    }
};
#endif

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(DTQW_TEST_NEON_ON_X86)
struct NEONBackend {
    using Vec4 = float32x4_t;

    static constexpr const char* name() noexcept { return "ARM NEON"; }

    static inline Vec4 setzero() noexcept { return vdupq_n_f32(0.0f); }
    static inline Vec4 set1(float v) noexcept { return vdupq_n_f32(v); }
    static inline Vec4 loadu(const float* p) noexcept { return vld1q_f32(p); }
    static inline void storeu(float* p, Vec4 v) noexcept { vst1q_f32(p, v); }

    static inline Vec4 add(Vec4 a, Vec4 b) noexcept { return vaddq_f32(a, b); }
    static inline Vec4 sub(Vec4 a, Vec4 b) noexcept { return vsubq_f32(a, b); }
    static inline Vec4 mul(Vec4 a, Vec4 b) noexcept { return vmulq_f32(a, b); }
    static inline Vec4 fma(Vec4 a, Vec4 b, Vec4 c) noexcept {
#if defined(__aarch64__) && !defined(DTQW_TEST_NEON_ON_X86)
        return vfmaq_f32(c, a, b);
#else
        return vmlaq_f32(c, a, b);
#endif
    }

    static inline Vec4 set(float x0, float x1, float x2, float x3) noexcept {
        alignas(16) const float vals[4] = { x0, x1, x2, x3 };
        return vld1q_f32(vals);
    }

    /// Parameterized unitary coin with continuous phase rotation theta:
    /// Eigenvalues rotate as exp(i*(pi -+ theta)) on the transversal scattering plane.
    /// theta = 0 corresponds to the standard Grover coin.
    static inline Vec4 coin(Vec4 v, Vec4 coeff_a, Vec4 coeff_odd, Vec4 coeff_even) noexcept {
        float32x4_t v_halves = vcombine_f32(vget_high_f32(v), vget_low_f32(v));
        float32x4_t diff     = vsubq_f32(v, v_halves);
        float32x4_t term_a   = vmulq_f32(coeff_a, diff);

        float32x2_t low_rev  = vrev64_f32(vget_low_f32(v));
        float32x4_t v_odd    = vcombine_f32(low_rev, low_rev);

        float32x2_t high_rev = vrev64_f32(vget_high_f32(v));
        float32x4_t v_even   = vcombine_f32(high_rev, high_rev);

#if defined(__aarch64__) && !defined(DTQW_TEST_NEON_ON_X86)
        float32x4_t term_bd  = vfmaq_f32(vmulq_f32(coeff_even, v_even), coeff_odd, v_odd);
#else
        float32x4_t term_bd  = vmlaq_f32(vmulq_f32(coeff_even, v_even), coeff_odd, v_odd);
#endif
        return vaddq_f32(term_bd, term_a);
    }

    /// Fast 4-port Grover scattering matrix C = (1/2)*1*1^T - I
    static inline Vec4 grover(Vec4 v) noexcept {
#if defined(__aarch64__) && !defined(DTQW_TEST_NEON_ON_X86)
        float sum = vaddvq_f32(v);
        float32x4_t half_sum = vdupq_n_f32(0.5f * sum);
        return vsubq_f32(half_sum, v);
#else
        float32x2_t low  = vget_low_f32(v);
        float32x2_t high = vget_high_f32(v);
        float32x2_t sum2 = vadd_f32(low, high);
        float32x2_t sum4 = vpadd_f32(sum2, sum2);
        float32x4_t half_sum = vmulq_n_f32(vcombine_f32(sum4, sum4), 0.5f);
        return vsubq_f32(half_sum, v);
#endif
    }

    static inline void transpose4x4(Vec4& r0, Vec4& r1, Vec4& r2, Vec4& r3) noexcept {
        float32x4x2_t z01 = vzipq_f32(r0, r1);
        float32x4x2_t z23 = vzipq_f32(r2, r3);
        r0 = vcombine_f32(vget_low_f32(z01.val[0]), vget_low_f32(z23.val[0]));
        r1 = vcombine_f32(vget_high_f32(z01.val[0]), vget_high_f32(z23.val[0]));
        r2 = vcombine_f32(vget_low_f32(z01.val[1]), vget_low_f32(z23.val[1]));
        r3 = vcombine_f32(vget_high_f32(z01.val[1]), vget_high_f32(z23.val[1]));
    }

    /// 4-point, 3rd-order Hermite interpolation across 4 lanes
    static inline Vec4 hermite(Vec4 ym1, Vec4 y0, Vec4 y1, Vec4 y2, Vec4 frac) noexcept {
        const Vec4 half     = vdupq_n_f32(0.5f);
        const Vec4 one_half = vdupq_n_f32(1.5f);
        const Vec4 two      = vdupq_n_f32(2.0f);
        const Vec4 two_half = vdupq_n_f32(2.5f);

        Vec4 c0 = y0;
        Vec4 c1 = mul(half, sub(y1, ym1));
        Vec4 c2 = sub(add(ym1, mul(two, y1)), add(mul(two_half, y0), mul(half, y2)));
        Vec4 c3 = add(mul(half, sub(y2, ym1)), mul(one_half, sub(y0, y1)));

        Vec4 acc = fma(c3, frac, c2);
        acc = fma(acc, frac, c1);
        return fma(acc, frac, c0);
    }

    /// Branchless phase wrapping for phase in [0, 2) to [0, 1)
    static inline Vec4 wrap_01(Vec4 p) noexcept {
        const Vec4 one = vdupq_n_f32(1.0f);
        uint32x4_t mask = vcgeq_f32(p, one);
        return vbslq_f32(mask, vsubq_f32(p, one), p);
    }

    /// Fast parabolic trigonometric approximation for normalized phase in [0, 1)
    /// Max error < 0.0011 (< 0.1% THD), branchless NEON
    static inline Vec4 fast_sin(Vec4 phase_01) noexcept {
        const Vec4 two   = vdupq_n_f32(2.0f);
        const Vec4 one   = vdupq_n_f32(1.0f);
        const Vec4 four  = vdupq_n_f32(4.0f);
        const Vec4 p0225 = vdupq_n_f32(0.225f);

        Vec4 x = vsubq_f32(vmulq_f32(two, phase_01), one); // [-1, 1)
        Vec4 abs_x = vabsq_f32(x);
        Vec4 y = vmulq_f32(four, vmulq_f32(x, vsubq_f32(one, abs_x)));
        Vec4 abs_y = vabsq_f32(y);
#if defined(__aarch64__) && !defined(DTQW_TEST_NEON_ON_X86)
        Vec4 y_refine = vfmaq_f32(y, vmulq_f32(p0225, y), vsubq_f32(abs_y, one));
#else
        Vec4 y_refine = vmlaq_f32(y, vmulq_f32(p0225, y), vsubq_f32(abs_y, one));
#endif
        return vnegq_f32(y_refine);
    }
};
#endif

struct ScalarBackend {
    using Vec4 = std::array<float, 4>;

    static constexpr const char* name() noexcept { return "Scalar"; }

    static inline Vec4 setzero() noexcept { return {0.0f, 0.0f, 0.0f, 0.0f}; }
    static inline Vec4 set1(float v) noexcept { return {v, v, v, v}; }
    static inline Vec4 loadu(const float* p) noexcept { return {p[0], p[1], p[2], p[3]}; }
    static inline void storeu(float* p, Vec4 v) noexcept { p[0] = v[0]; p[1] = v[1]; p[2] = v[2]; p[3] = v[3]; }

    static inline Vec4 add(Vec4 a, Vec4 b) noexcept { return {a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]}; }
    static inline Vec4 sub(Vec4 a, Vec4 b) noexcept { return {a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]}; }
    static inline Vec4 mul(Vec4 a, Vec4 b) noexcept { return {a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]}; }
    static inline Vec4 fma(Vec4 a, Vec4 b, Vec4 c) noexcept {
        return {std::fma(a[0], b[0], c[0]), std::fma(a[1], b[1], c[1]),
                std::fma(a[2], b[2], c[2]), std::fma(a[3], b[3], c[3])};
    }

    static inline Vec4 set(float x0, float x1, float x2, float x3) noexcept {
        return {x0, x1, x2, x3};
    }

    static inline Vec4 coin(Vec4 v, Vec4 coeff_a, Vec4 coeff_odd, Vec4 coeff_even) noexcept {
        const float a = coeff_a[0];
        const float b = coeff_odd[0];
        const float d = coeff_even[0];

        const float d02 = v[0] - v[2];
        const float d13 = v[1] - v[3];

        return {
            b * v[1] + d * v[3] + a * d02,
            d * v[0] + b * v[2] + a * d13,
            d * v[1] + b * v[3] - a * d02,
            b * v[0] + d * v[2] - a * d13
        };
    }

    static inline Vec4 grover(Vec4 v) noexcept {
        const float half_sum = 0.5f * (v[0] + v[1] + v[2] + v[3]);
        return {half_sum - v[0], half_sum - v[1], half_sum - v[2], half_sum - v[3]};
    }

    static inline void transpose4x4(Vec4& r0, Vec4& r1, Vec4& r2, Vec4& r3) noexcept {
        Vec4 t0 = {r0[0], r1[0], r2[0], r3[0]};
        Vec4 t1 = {r0[1], r1[1], r2[1], r3[1]};
        Vec4 t2 = {r0[2], r1[2], r2[2], r3[2]};
        Vec4 t3 = {r0[3], r1[3], r2[3], r3[3]};
        r0 = t0; r1 = t1; r2 = t2; r3 = t3;
    }

    static inline Vec4 hermite(Vec4 ym1, Vec4 y0, Vec4 y1, Vec4 y2, Vec4 frac) noexcept {
        const Vec4 half     = set1(0.5f);
        const Vec4 one_half = set1(1.5f);
        const Vec4 two      = set1(2.0f);
        const Vec4 two_half = set1(2.5f);

        Vec4 c0 = y0;
        Vec4 c1 = mul(half, sub(y1, ym1));
        Vec4 c2 = sub(add(ym1, mul(two, y1)), add(mul(two_half, y0), mul(half, y2)));
        Vec4 c3 = add(mul(half, sub(y2, ym1)), mul(one_half, sub(y0, y1)));

        Vec4 acc = fma(c3, frac, c2);
        acc = fma(acc, frac, c1);
        return fma(acc, frac, c0);
    }

    static inline Vec4 wrap_01(Vec4 p) noexcept {
        Vec4 res = p;
        for (int i = 0; i < 4; ++i) {
            if (res[i] >= 1.0f) res[i] -= 1.0f;
        }
        return res;
    }

    static inline Vec4 fast_sin(Vec4 phase_01) noexcept {
        Vec4 res{};
        for (int i = 0; i < 4; ++i) {
            float x = 2.0f * phase_01[i] - 1.0f;
            float abs_x = std::abs(x);
            float y = 4.0f * x * (1.0f - abs_x);
            float abs_y = std::abs(y);
            float y_refine = 0.225f * y * (abs_y - 1.0f) + y;
            res[i] = -y_refine;
        }
        return res;
    }
};

#if defined(__ARM_NEON) || defined(__aarch64__)
  using DefaultBackend = NEONBackend;
#elif defined(__SSE__) || defined(_M_X64)
  using DefaultBackend = SSEBackend;
#else
  using DefaultBackend = ScalarBackend;
#endif

#if !defined(__SSE__) && !defined(_M_X64)
  using SSEBackend = DefaultBackend;
#endif

#if !defined(__ARM_NEON) && !defined(__aarch64__)
  #if !defined(DTQW_TEST_NEON_ON_X86)
    using NEONBackend = DefaultBackend;
  #endif
#endif

} // namespace dtqw::simd

namespace dtqw {

/// Delay-Line Bank mit 4-Sample Guard-Zone am Ringpuffer-Ende für zusammenhängende SIMD-Loads
template <typename SimdBackend, size_t NumLines = 64, size_t BufferOrder = 13>
class DTQWDelayLineBank {
public:
    using Vec4 = typename SimdBackend::Vec4;

    static constexpr size_t kBufferSize   = 1ULL << BufferOrder; // 8192 Samples pro Leitung
    static constexpr size_t kBufferMask   = kBufferSize - 1;
    static constexpr size_t kGuardSamples = 4;
    static constexpr size_t kAllocSize    = kBufferSize + kGuardSamples;
    static constexpr size_t kTotalFloats  = NumLines * kAllocSize;

    DTQWDelayLineBank() {
        memory_.resize(kTotalFloats, 0.0f);
        write_pos_ = 0;
    }

    void reset() noexcept {
        std::fill(memory_.begin(), memory_.end(), 0.0f);
        write_pos_ = 0;
    }

    inline void write(size_t line_idx, float sample) noexcept {
        float* buf = &memory_[line_idx * kAllocSize];
        buf[write_pos_] = sample;
        if (write_pos_ < kGuardSamples) {
            buf[kBufferSize + write_pos_] = sample;
        }
    }

    inline void advance() noexcept {
        write_pos_ = (write_pos_ + 1) & kBufferMask;
    }

    inline Vec4 readNodePorts(size_t start_line_idx, 
                             const std::array<float, 4>& delays) const noexcept 
    {
        alignas(16) std::array<float, 4> fracs;
        alignas(16) std::array<int32_t, 4> base_indices;

        for (int p = 0; p < 4; ++p) {
            const float d = delays[p];
            const float d_floor = std::floor(d);
            const int32_t d_int = static_cast<int32_t>(d_floor);
            fracs[p] = d - d_floor;
            base_indices[p] = static_cast<int32_t>((write_pos_ - d_int - 2) & kBufferMask);
        }

        const float* b0 = &memory_[(start_line_idx + 0) * kAllocSize];
        const float* b1 = &memory_[(start_line_idx + 1) * kAllocSize];
        const float* b2 = &memory_[(start_line_idx + 2) * kAllocSize];
        const float* b3 = &memory_[(start_line_idx + 3) * kAllocSize];

        Vec4 row0 = SimdBackend::loadu(&b0[base_indices[0]]);
        Vec4 row1 = SimdBackend::loadu(&b1[base_indices[1]]);
        Vec4 row2 = SimdBackend::loadu(&b2[base_indices[2]]);
        Vec4 row3 = SimdBackend::loadu(&b3[base_indices[3]]);

        SimdBackend::transpose4x4(row0, row1, row2, row3);
        const Vec4 frac_vec = SimdBackend::loadu(fracs.data());
        return SimdBackend::hermite(row3, row2, row1, row0, frac_vec);
    }

private:
    std::vector<float> memory_;
    size_t write_pos_{0};
};

/// Vektorisierte 4-Kanal 1-Pol TPT (Topology-Preserving Transform) Dämpfungsfilterbank
template <typename SimdBackend>
class TPTDampingFilter4 {
public:
    using Vec4 = typename SimdBackend::Vec4;

    TPTDampingFilter4() noexcept { reset(); }

    void reset() noexcept {
        state_ = SimdBackend::setzero();
        G_     = SimdBackend::set1(1.0f);
        gain_  = SimdBackend::set1(1.0f);
    }

    void setParameters(const std::array<float, 4>& cutoff_hz,
                       const std::array<float, 4>& loop_gains,
                       double sample_rate) noexcept
    {
        alignas(16) std::array<float, 4> G_arr;
        const float fs = static_cast<float>(sample_rate);
        for (int i = 0; i < 4; ++i) {
            float g = std::tan(static_cast<float>(M_PI) * cutoff_hz[i] / fs);
            g = std::max(1e-5f, std::min(g, 100.0f));
            G_arr[i] = g / (1.0f + g);
        }
        G_    = SimdBackend::loadu(G_arr.data());
        gain_ = SimdBackend::loadu(loop_gains.data());
    }

    inline Vec4 process(Vec4 x) noexcept {
        Vec4 delta = SimdBackend::sub(x, state_);
        Vec4 v     = SimdBackend::mul(G_, delta);
        Vec4 y_lp  = SimdBackend::add(v, state_);
        const Vec4 two = SimdBackend::set1(2.0f);
        state_         = SimdBackend::fma(two, v, state_);
        return SimdBackend::mul(gain_, y_lp);
    }

private:
    Vec4 state_;
    Vec4 G_;
    Vec4 gain_;
};

/// Vektorisierte 4-Kanal Allpass-Kaskade zur frequenzabhängigen Kanten-Dispersion
/// Jede der 4 Kanten/Ports durchläuft NumStages 1-Pol Allpassfilter in 1-Multiplizierer-Gitterstruktur:
///   y_s[n] = state_s[n-1] - g_s * x_s[n]
///   state_s[n] = x_s[n] + g_s * y_s[n]
///
/// Übertragungsfunktion pro Stufe: A(z) = (-g + z^-1) / (1 - g * z^-1)
/// Betragsfrequenzgang: |A(e^jw)| = 1.0 (exakt verlustlos und unitär)
/// Gruppenlaufzeit: tau_g(w) = (1 - g^2) / (1 + g^2 - 2g cos w)
///   - g > 0: tau_g(0) > tau_g(pi) (Tiefen verzögert, Höhen eilen voraus / Up-Chirp)
///   - g < 0: tau_g(0) < tau_g(pi) (Höhen verzögert, Tiefen eilen voraus / Down-Chirp)
///   - g = 0: tau_g(w) = 1 Sample pro Stufe (reine Laufzeit ohne Dispersion)
template <typename SimdBackend, size_t NumStages = 2>
class AllpassCascade4 {
public:
    using Vec4 = typename SimdBackend::Vec4;

    AllpassCascade4() noexcept { reset(); }

    void reset() noexcept {
        for (size_t s = 0; s < NumStages; ++s) {
            state0_[s] = SimdBackend::setzero();
            state1_[s] = SimdBackend::setzero();
            g0_[s]     = SimdBackend::setzero();
            g1_[s]     = SimdBackend::setzero();
        }
        active_bank_    = 0;
        is_crossfading_ = false;
        crossfade_pos_  = 0.0f;
        crossfade_step_ = 0.0f;
    }

    void setCoefficients(Vec4 g) noexcept {
        for (size_t s = 0; s < NumStages; ++s) {
            g0_[s] = g;
            g1_[s] = g;
        }
        is_crossfading_ = false;
    }

    void setCoefficientsForStage(size_t stage, Vec4 g) noexcept {
        if (stage < NumStages) {
            g0_[stage] = g;
            g1_[stage] = g;
        }
    }

    void setCoefficient(float g) noexcept {
        Vec4 vg = SimdBackend::set1(g);
        for (size_t s = 0; s < NumStages; ++s) {
            g0_[s] = vg;
            g1_[s] = vg;
        }
        is_crossfading_ = false;
    }

    /// Startet einen dualen zustandsübergreifenden Crossfade zu neuen Koeffizienten
    void startCrossfade(Vec4 new_g, size_t num_samples) noexcept {
        if (num_samples <= 1) {
            setCoefficients(new_g);
            return;
        }

        if (is_crossfading_) {
            // Überlappender Crossfade: aktuelle interpolierte Zustände und Koeffizienten glatt als neue Basis übernehmen
            float pos = crossfade_pos_;
            float p = pos * pos * (3.0f - 2.0f * pos);
            Vec4 vp = SimdBackend::set1(p);
            Vec4 v_inv_p = SimdBackend::set1(1.0f - p);

            for (size_t s = 0; s < NumStages; ++s) {
                Vec4 from_state = (active_bank_ == 0) ? state0_[s] : state1_[s];
                Vec4 to_state   = (active_bank_ == 0) ? state1_[s] : state0_[s];
                Vec4 from_g     = (active_bank_ == 0) ? g0_[s] : g1_[s];
                Vec4 to_g       = (active_bank_ == 0) ? g1_[s] : g0_[s];

                Vec4 s_blend = SimdBackend::add(SimdBackend::mul(v_inv_p, from_state), SimdBackend::mul(vp, to_state));
                Vec4 g_blend = SimdBackend::add(SimdBackend::mul(v_inv_p, from_g), SimdBackend::mul(vp, to_g));

                state0_[s] = s_blend;
                state1_[s] = s_blend;
                g0_[s]     = g_blend;
                g1_[s]     = new_g;
            }
            active_bank_    = 0;
            is_crossfading_ = true;
            crossfade_pos_  = 0.0f;
            crossfade_step_ = 1.0f / static_cast<float>(num_samples);
            return;
        }

        int target_bank = 1 - active_bank_;
        Vec4* active_state = (active_bank_ == 0) ? state0_ : state1_;
        Vec4* target_state = (target_bank == 0) ? state0_ : state1_;
        Vec4* target_g     = (target_bank == 0) ? g0_ : g1_;

        for (size_t s = 0; s < NumStages; ++s) {
            target_g[s] = new_g;
            // Zustandsübertragung: Ziel-Bank erbt den momentanen Filterzustand
            target_state[s] = active_state[s];
        }

        is_crossfading_ = true;
        crossfade_pos_  = 0.0f;
        crossfade_step_ = 1.0f / static_cast<float>(num_samples);
    }

    void startCrossfade(float new_g, size_t num_samples) noexcept {
        startCrossfade(SimdBackend::set1(new_g), num_samples);
    }

    bool isCrossfading() const noexcept { return is_crossfading_; }
    float getCrossfadePosition() const noexcept { return crossfade_pos_; }
    int getActiveBank() const noexcept { return active_bank_; }

    inline Vec4 process(Vec4 x) noexcept {
        if (!is_crossfading_) {
            Vec4* st = (active_bank_ == 0) ? state0_ : state1_;
            Vec4* g  = (active_bank_ == 0) ? g0_ : g1_;
            Vec4 current = x;
            for (size_t s = 0; s < NumStages; ++s) {
                Vec4 gx = SimdBackend::mul(g[s], current);
                Vec4 y  = SimdBackend::sub(st[s], gx);
                st[s]   = SimdBackend::fma(g[s], y, current);
                current = y;
            }
            return current;
        }

        // Parallele Ausführung beider Filterbänke während des Zustands-Crossfades
        Vec4 cur0 = x;
        for (size_t s = 0; s < NumStages; ++s) {
            Vec4 gx = SimdBackend::mul(g0_[s], cur0);
            Vec4 y  = SimdBackend::sub(state0_[s], gx);
            state0_[s] = SimdBackend::fma(g0_[s], y, cur0);
            cur0 = y;
        }

        Vec4 cur1 = x;
        for (size_t s = 0; s < NumStages; ++s) {
            Vec4 gx = SimdBackend::mul(g1_[s], cur1);
            Vec4 y  = SimdBackend::sub(state1_[s], gx);
            state1_[s] = SimdBackend::fma(g1_[s], y, cur1);
            cur1 = y;
        }

        // C1-glatte Smoothstep-Interpolation (Null-Ableitung an den Rändern)
        float pos = crossfade_pos_;
        float p = pos * pos * (3.0f - 2.0f * pos);
        Vec4 vp = SimdBackend::set1(p);
        Vec4 v_inv_p = SimdBackend::set1(1.0f - p);

        Vec4 out = (active_bank_ == 0)
            ? SimdBackend::add(SimdBackend::mul(v_inv_p, cur0), SimdBackend::mul(vp, cur1))
            : SimdBackend::add(SimdBackend::mul(v_inv_p, cur1), SimdBackend::mul(vp, cur0));

        crossfade_pos_ += crossfade_step_;
        if (crossfade_pos_ >= 1.0f) {
            crossfade_pos_  = 1.0f;
            active_bank_    = 1 - active_bank_;
            is_crossfading_ = false;
        }

        return out;
    }

private:
    alignas(16) Vec4 state0_[NumStages]{};
    alignas(16) Vec4 state1_[NumStages]{};
    alignas(16) Vec4 g0_[NumStages]{};
    alignas(16) Vec4 g1_[NumStages]{};
    int active_bank_{0};
    bool is_crossfading_{false};
    float crossfade_pos_{0.0f};
    float crossfade_step_{0.0f};
};

/// DTQW Hypercube Knoten mit parametrisiertem unitären Coin, TPT-Dämpfung und Allpass-Dispersion
template <typename SimdBackend, size_t NumAllpassStages = 2>
class DTQWNode {
public:
    using Vec4 = typename SimdBackend::Vec4;

    DTQWNode() noexcept { reset(); }

    void reset() noexcept {
        filter_.reset();
        dispersion_filter_.reset();
        // Default: theta = 0 (Grover Diffuser: a = -0.5, b = 0.5, d = 0.5)
        coeff_a_    = SimdBackend::set1(-0.5f);
        coeff_odd_  = SimdBackend::set(0.5f, 0.5f, 0.5f, 0.5f);
        coeff_even_ = SimdBackend::set(0.5f, 0.5f, 0.5f, 0.5f);
        dispersion_enabled_ = false;
        dispersion_mix_ = 0.0f;
    }

    void setDamping(const std::array<float, 4>& cutoffs,
                    const std::array<float, 4>& gains,
                    double sample_rate) noexcept
    {
        filter_.setParameters(cutoffs, gains, sample_rate);
    }

    void setCoinCoefficients(Vec4 a, Vec4 odd, Vec4 even) noexcept {
        coeff_a_    = a;
        coeff_odd_  = odd;
        coeff_even_ = even;
    }

    void setDispersionCoefficients(Vec4 g) noexcept {
        dispersion_filter_.setCoefficients(g);
    }

    void startDispersionCrossfade(Vec4 new_g, size_t num_samples) noexcept {
        dispersion_filter_.startCrossfade(new_g, num_samples);
    }

    bool isDispersionCrossfading() const noexcept {
        return dispersion_filter_.isCrossfading();
    }

    void setDispersionEnabled(bool enabled) noexcept {
        dispersion_enabled_ = enabled;
        dispersion_mix_ = enabled ? 1.0f : 0.0f;
    }

    void setDispersionMix(float mix) noexcept {
        dispersion_mix_ = std::max(0.0f, std::min(mix, 1.0f));
        dispersion_enabled_ = (dispersion_mix_ > 0.0f);
    }

    bool isDispersionEnabled() const noexcept {
        return dispersion_enabled_;
    }

    float getDispersionMix() const noexcept {
        return dispersion_mix_;
    }

    inline Vec4 process(Vec4 incoming_ports, Vec4 injection = SimdBackend::setzero()) noexcept {
        Vec4 total_in = SimdBackend::add(incoming_ports, injection);
        Vec4 scattered = SimdBackend::coin(total_in, coeff_a_, coeff_odd_, coeff_even_);
        Vec4 damped = filter_.process(scattered);
        if (dispersion_mix_ <= 0.0f) {
            return damped;
        }
        Vec4 dispersed = dispersion_filter_.process(damped);
        if (dispersion_mix_ >= 1.0f) {
            return dispersed;
        }
        // Klickfreier linearer Crossfade während des Umschaltens
        Vec4 v_mix = SimdBackend::set1(dispersion_mix_);
        Vec4 v_inv = SimdBackend::set1(1.0f - dispersion_mix_);
        return SimdBackend::add(SimdBackend::mul(v_inv, damped), SimdBackend::mul(v_mix, dispersed));
    }

private:
    TPTDampingFilter4<SimdBackend> filter_;
    AllpassCascade4<SimdBackend, NumAllpassStages> dispersion_filter_;
    Vec4 coeff_a_{};
    Vec4 coeff_odd_{};
    Vec4 coeff_even_{};
    bool dispersion_enabled_{false};
    float dispersion_mix_{0.0f};
};

/// 32-Kanal LFO Bank zur reziproken Delay-Kantenmodulation
template <typename SimdBackend>
class DTQWLfoBank32 {
public:
    using Vec4 = typename SimdBackend::Vec4;
    static constexpr size_t kNumLFOs = 32;

    DTQWLfoBank32() noexcept {
        reset();
    }

    void reset() noexcept {
        constexpr float kGolden = 0.618033988749895f;
        for (size_t i = 0; i < kNumLFOs; ++i) {
            phases_[i] = std::fmod(static_cast<float>(i) * kGolden, 1.0f);
            outputs_[i] = 0.0f;
        }
    }

    void prepare(float rate_hz, double sample_rate) noexcept {
        sample_rate_ = sample_rate;
        setRate(rate_hz);
    }

    void setRate(float rate_hz) noexcept {
        rate_hz_ = std::max(0.01f, std::min(rate_hz, 20.0f));
        const float fs = static_cast<float>(sample_rate_);
        for (size_t i = 0; i < kNumLFOs; ++i) {
            // Deterministische Frequenz-Dispersion (+- 7%) verhindert stehende Interferenzmoden
            float spread = 1.0f + 0.07f * std::sin(static_cast<float>(i) * 0.39269908f);
            phase_increments_[i] = (rate_hz_ * spread) / fs;
        }
    }

    /// Berechnet 32 LFO-Modulationen für das aktuelle Sample
    inline void step(float current_depth_samples) noexcept {
        const Vec4 v_depth = SimdBackend::set1(current_depth_samples);

        for (size_t i = 0; i < kNumLFOs; i += 4) {
            Vec4 p = SimdBackend::loadu(&phases_[i]);
            Vec4 dp = SimdBackend::loadu(&phase_increments_[i]);
            p = SimdBackend::wrap_01(SimdBackend::add(p, dp));
            SimdBackend::storeu(&phases_[i], p);

            Vec4 s = SimdBackend::fast_sin(p);
            Vec4 mod = SimdBackend::mul(s, v_depth);
            SimdBackend::storeu(&outputs_[i], mod);
        }
    }

    inline float getModulation(size_t edge_idx) const noexcept {
        return outputs_[edge_idx];
    }

    const float* data() const noexcept { return outputs_.data(); }

private:
    double sample_rate_{48000.0};
    float rate_hz_{0.8f};
    alignas(16) std::array<float, kNumLFOs> phases_{};
    alignas(16) std::array<float, kNumLFOs> phase_increments_{};
    alignas(16) std::array<float, kNumLFOs> outputs_{};
};

/// Vollständige 4D-Hypercube DTQW Reverb Engine mit 32 LFOs und parametergeglätteter Raumgrößenskalierung
template <typename SimdBackend = simd::DefaultBackend, size_t NumAllpassStages = 2>
class DTQWEngine {
public:
    using Vec4 = typename SimdBackend::Vec4;

    static constexpr size_t kNumNodes   = 16;
    static constexpr size_t kNumPorts   = 4;
    static constexpr size_t kTotalLines = kNumNodes * kNumPorts; // 64 gerichtete Kanten
    static constexpr size_t kNumEdges   = 32;                   // 32 ungerichtete Kanten

    alignas(16) static constexpr float kHadamard16[16][16] = {
        {  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f }, // Row  0
        {  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f }, // Row  1
        {  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f }, // Row  2
        {  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f }, // Row  3
        {  0.2500f,  0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f }, // Row  4
        {  0.2500f, -0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f }, // Row  5
        {  0.2500f,  0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f }, // Row  6
        {  0.2500f, -0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f }, // Row  7
        {  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f }, // Row  8
        {  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f }, // Row  9
        {  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f }, // Row 10
        {  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f }, // Row 11
        {  0.2500f,  0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f }, // Row 12
        {  0.2500f, -0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f }, // Row 13
        {  0.2500f,  0.2500f, -0.2500f, -0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f }, // Row 14
        {  0.2500f, -0.2500f, -0.2500f,  0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f,  0.2500f, -0.2500f,  0.2500f, -0.2500f, -0.2500f,  0.2500f } // Row 15
    };


    DTQWEngine() {
        initEdgeTopology();
        prepare(48000.0);
    }

    void prepare(double sample_rate) {
        sample_rate_ = sample_rate;
        delay_bank_.reset();
        for (auto& node : nodes_) {
            node.reset();
        }
        lfo_bank_.prepare(chorus_rate_hz_, sample_rate_);

        updateSmoothingCoefficients();
        updateCoinCoefficients();
        updateDispersionCoefficients();
        recalculateGainsAndDelays(true);
    }

    void reset() noexcept {
        delay_bank_.reset();
        for (auto& node : nodes_) {
            node.reset();
        }
        lfo_bank_.reset();
        current_room_size_ = target_room_size_;
        current_chorus_depth_ = target_chorus_depth_;
        current_coin_phase_ = target_coin_phase_;
        current_dispersion_ = target_dispersion_;
        current_dispersion_ports_ = target_dispersion_ports_;
        dispersion_start_ports_ = target_dispersion_ports_;
        current_dispersion_mix_ = dispersion_enabled_ ? 1.0f : 0.0f;
        target_dispersion_mix_ = current_dispersion_mix_;
        dispersion_is_ramping_ = false;
        dispersion_ramp_counter_ = 0;
        dispersion_ramp_samples_ = 0;
        dispersion_ramp_step_.fill(0.0f);
        dispersion_crossfade_active_ = false;
        dispersion_crossfade_counter_ = 0;
        updateCoinCoefficients();
        updateDispersionCoefficients();
        for (auto& node : nodes_) {
            node.setDispersionMix(current_dispersion_mix_);
        }
        gain_update_counter_ = 0;
        last_gain_room_size_ = current_room_size_;
        recalculateGainsAndDelays(true);
    }

    // ========================================================================
    // Parameter Control Interface (Voll steuerbar)
    // ========================================================================

    /// Dynamische Raumgröße (Skalierungsfaktor 0.1x bis 3.5x)
    void setRoomSize(float size, bool immediate = false) noexcept {
        target_room_size_ = std::max(0.1f, std::min(size, 3.5f));
        if (immediate) {
            current_room_size_ = target_room_size_;
            recalculateGainsAndDelays(true);
        }
    }

    /// Glättungszeit für Raumgrößen-Änderungen in Millisekunden (1 ms bis 2000 ms)
    void setRoomSizeSmoothingTime(float time_ms) noexcept {
        room_size_smooth_ms_ = std::max(1.0f, std::min(time_ms, 2000.0f));
        updateSmoothingCoefficients();
    }

    /// Modulationsgeschwindigkeit des Chorusing in Hz (0.01 Hz bis 15.0 Hz)
    void setChorusRate(float rate_hz) noexcept {
        chorus_rate_hz_ = std::max(0.01f, std::min(rate_hz, 15.0f));
        lfo_bank_.setRate(chorus_rate_hz_);
    }

    /// Modulationstiefe des Chorusing in Samples (0.0 bis 24.0 Samples)
    void setChorusDepth(float depth_samples, bool immediate = false) noexcept {
        target_chorus_depth_ = std::max(0.0f, std::min(depth_samples, 24.0f));
        if (immediate) {
            current_chorus_depth_ = target_chorus_depth_;
        }
    }

    /// Nachhallzeit T60 in Sekunden (0.1 s bis 60.0 s)
    void setDecayTime(float t60_seconds) noexcept {
        t60_seconds_ = std::max(0.1f, std::min(t60_seconds, 60.0f));
        recalculateGainsAndDelays(true);
    }

    /// Dämpfungs-Grenzfrequenz in Hz (200 Hz bis Nyquist)
    void setDampingCutoff(float cutoff_hz) noexcept {
        damping_cutoff_hz_ = std::max(200.0f, std::min(cutoff_hz, static_cast<float>(sample_rate_ * 0.49)));
        recalculateGainsAndDelays(true);
    }

    /// Dry/Wet Mischungsverhältnis (0.0 = Dry, 1.0 = Wet)
    void setDryWet(float wet_fraction) noexcept {
        wet_fraction_ = std::max(0.0f, std::min(wet_fraction, 1.0f));
    }

    /// Konfiguriert die Walsh-Hadamard-Zeilen für Injektion und Auskopplung (0..15)
    void setHadamardRouting(int in_l, int in_r, int out_l, int out_r) noexcept {
        in_l_row_ = in_l & 15;
        in_r_row_ = in_r & 15;
        out_l_row_ = out_l & 15;
        out_r_row_ = out_r & 15;
    }

    /// Setzt das Hadamard-Routing auf das optimale isomorphe Default-Paar zurück:
    /// In: Dipole (1, 2) [Bit 0, Bit 1], Out: Quadrupole (5, 6) [Bit 0+2, Bit 1+2]
    void resetHadamardRoutingToDefault() noexcept {
        in_l_row_ = 1;
        in_r_row_ = 2;
        out_l_row_ = 5;
        out_r_row_ = 6;
    }

    int getHadamardInL() const noexcept { return in_l_row_; }
    int getHadamardInR() const noexcept { return in_r_row_; }
    int getHadamardOutL() const noexcept { return out_l_row_; }
    int getHadamardOutR() const noexcept { return out_r_row_; }

    /// Stufenlose unitäre Coin-Phasenrotation theta in Radians [0, 2*pi).
    /// theta = 0 entspricht exakt dem klassischen Grover-Diffusor.
    /// theta rotiert stufenlos die komplexen Eigenwerte der transversalen Streuebene (e2, e3)
    /// als exp(i * (pi -+ theta)), unter strikter Unitarität (C^T * C = I) und
    /// vollständiger Erhaltung des Injektions-Eigenraums (Null-Durchschlag bei t=0).
    void setCoinPhase(float phase_radians, bool immediate = false) noexcept {
        target_coin_phase_ = phase_radians;
        if (immediate) {
            current_coin_phase_ = target_coin_phase_;
            updateCoinCoefficients();
        }
    }

    /// Glättungszeit für Phasenrotations-Änderungen in Millisekunden (1 ms bis 2000 ms)
    void setCoinPhaseSmoothingTime(float time_ms) noexcept {
        coin_phase_smooth_ms_ = std::max(1.0f, std::min(time_ms, 2000.0f));
        updateSmoothingCoefficients();
    }

    float getCoinPhase() const noexcept { return target_coin_phase_; }
    float getCurrentCoinPhase() const noexcept { return current_coin_phase_; }
    float getCoinPhaseSmoothingTime() const noexcept { return coin_phase_smooth_ms_; }

    // ========================================================================
    // Dispersion Parameter Control Interface (Allpass-Kaskaden in den Kanten)
    // ========================================================================

    /// Dispersionsgrad g in [-0.95, +0.95] (mit geglätteter Host-Automation)
    /// Positive Werte: Tiefen verzögert, Höhen eilen voraus (akustischer Up-Chirp / Saitensteifigkeit)
    /// Negative Werte: Höhen verzögert, Tiefen eilen voraus (akustischer Down-Chirp)
    /// g = 0.0: keine frequenzabhängige Dispersion
    void startDispersionStateCrossfade(const std::array<float, 4>& target_ports) noexcept {
        dispersion_start_ports_ = current_dispersion_ports_;
        target_dispersion_ports_ = target_ports;
        target_dispersion_ = target_dispersion_ports_[0];

        const float fs = static_cast<float>(sample_rate_);
        size_t crossfade_samples = std::max<size_t>(16, static_cast<size_t>((dispersion_crossfade_ms_ * 0.001f) * fs));
        dispersion_crossfade_samples_ = crossfade_samples;
        dispersion_crossfade_counter_ = 0;
        dispersion_crossfade_active_  = true;

        Vec4 new_g = SimdBackend::loadu(target_dispersion_ports_.data());
        for (auto& node : nodes_) {
            node.startDispersionCrossfade(new_g, crossfade_samples);
        }
    }

    void setDispersion(float amount, bool immediate = false) noexcept {
        target_dispersion_ = std::max(-0.95f, std::min(amount, 0.95f));
        target_dispersion_ports_.fill(target_dispersion_);
        dispersion_is_ramping_ = false;
        dispersion_enabled_ = true;
        target_dispersion_mix_ = 1.0f;

        if (immediate) {
            current_dispersion_ = target_dispersion_;
            current_dispersion_ports_ = target_dispersion_ports_;
            current_dispersion_mix_ = 1.0f;
            dispersion_crossfade_active_ = false;
            updateDispersionCoefficients();
            for (auto& node : nodes_) {
                node.setDispersionMix(1.0f);
            }
        } else if (dispersion_crossfade_enabled_) {
            startDispersionStateCrossfade(target_dispersion_ports_);
        }
    }

    float getDispersion() const noexcept { return target_dispersion_; }
    float getCurrentDispersion() const noexcept { return current_dispersion_; }

    /// Glättungszeit für Dispersionsänderungen in Millisekunden (1 ms bis 2000 ms)
    void setDispersionSmoothingTime(float time_ms) noexcept {
        dispersion_smooth_ms_ = std::max(1.0f, std::min(time_ms, 2000.0f));
        dispersion_crossfade_ms_ = dispersion_smooth_ms_;
        updateSmoothingCoefficients();
    }

    float getDispersionSmoothingTime() const noexcept { return dispersion_smooth_ms_; }

    /// Dauer des zustandsübergreifenden Crossfades beim Ändern der Allpass-Koeffizienten in Millisekunden
    void setDispersionCrossfadeTime(float time_ms) noexcept {
        dispersion_crossfade_ms_ = std::max(1.0f, std::min(time_ms, 500.0f));
    }

    float getDispersionCrossfadeTime() const noexcept { return dispersion_crossfade_ms_; }

    /// Aktiviert oder deaktiviert den dualen Zustands-Crossfade (Default: true)
    void setDispersionCrossfadeEnabled(bool enabled) noexcept {
        dispersion_crossfade_enabled_ = enabled;
    }

    bool isDispersionCrossfadeEnabled() const noexcept { return dispersion_crossfade_enabled_; }

    /// Prüft, ob aktuell ein zustandsübergreifender Crossfade aktiv ist
    bool isDispersionCrossfading() const noexcept {
        return dispersion_crossfade_active_ || (nodes_.size() > 0 && nodes_[0].isDispersionCrossfading());
    }

    /// Schaltet die Allpass-Kaskaden in den Kanten klickfrei ein oder aus (mit geglättetem Crossfade)
    void setDispersionEnabled(bool enabled, bool immediate = false) noexcept {
        dispersion_enabled_ = enabled;
        target_dispersion_mix_ = enabled ? 1.0f : 0.0f;
        if (immediate) {
            current_dispersion_mix_ = target_dispersion_mix_;
            for (auto& node : nodes_) {
                node.setDispersionMix(current_dispersion_mix_);
            }
        }
    }

    bool isDispersionEnabled() const noexcept { return dispersion_enabled_; }
    float getDispersionMix() const noexcept { return current_dispersion_mix_; }

    /// Akustisch-frequenzbasierte Dispersionssteuerung: Berechnet den Allpass-Koeffizienten g
    /// über die bilineare Transformation so, dass bei freq_hz eine Phasendrehung von -pi/2 auftritt
    void setDispersionFrequency(float freq_hz, bool immediate = false) noexcept {
        const float fs = static_cast<float>(sample_rate_);
        float f = std::max(20.0f, std::min(freq_hz, fs * 0.48f));
        float omega = static_cast<float>(M_PI) * f / fs;
        float t = std::tan(omega);
        float g = (1.0f - t) / (1.0f + t);
        setDispersion(g, immediate);
    }

    /// Setzt individuelle Dispersionskoeffizienten für die 4 Dimensionen/Ports (mit geglättetem Übergang)
    void setPortDispersion(const std::array<float, 4>& g_ports, bool immediate = false) noexcept {
        for (size_t i = 0; i < 4; ++i) {
            target_dispersion_ports_[i] = std::max(-0.95f, std::min(g_ports[i], 0.95f));
        }
        target_dispersion_ = target_dispersion_ports_[0];
        dispersion_is_ramping_ = false;
        dispersion_enabled_ = true;
        target_dispersion_mix_ = 1.0f;

        if (immediate) {
            current_dispersion_ports_ = target_dispersion_ports_;
            current_dispersion_ = target_dispersion_;
            current_dispersion_mix_ = 1.0f;
            dispersion_crossfade_active_ = false;
            updateDispersionCoefficients();
            for (auto& node : nodes_) {
                node.setDispersionMix(1.0f);
            }
        } else if (dispersion_crossfade_enabled_) {
            startDispersionStateCrossfade(target_dispersion_ports_);
        }
    }

    /// Setzt den Dispersionskoeffizienten für einen einzelnen Port k in [0..3]
    void setPortDispersion(size_t port, float amount, bool immediate = false) noexcept {
        if (port >= 4) return;
        target_dispersion_ports_[port] = std::max(-0.95f, std::min(amount, 0.95f));
        target_dispersion_ = target_dispersion_ports_[0];
        dispersion_is_ramping_ = false;
        dispersion_enabled_ = true;
        target_dispersion_mix_ = 1.0f;

        if (immediate) {
            current_dispersion_ports_[port] = target_dispersion_ports_[port];
            current_dispersion_ = current_dispersion_ports_[0];
            current_dispersion_mix_ = 1.0f;
            dispersion_crossfade_active_ = false;
            updateDispersionCoefficients();
            for (auto& node : nodes_) {
                node.setDispersionMix(1.0f);
            }
        } else if (dispersion_crossfade_enabled_) {
            startDispersionStateCrossfade(target_dispersion_ports_);
        }
    }

    /// Setzt die Frequenz für einen einzelnen Port k in [0..3]
    void setPortDispersionFrequency(size_t port, float freq_hz, bool immediate = false) noexcept {
        if (port >= 4) return;
        const float fs = static_cast<float>(sample_rate_);
        float f = std::max(20.0f, std::min(freq_hz, fs * 0.48f));
        float omega = static_cast<float>(M_PI) * f / fs;
        float t = std::tan(omega);
        float g = (1.0f - t) / (1.0f + t);
        setPortDispersion(port, g, immediate);
    }

    /// Sample-genaue lineare Host-Automation: Fährt Dispersion über ramp_samples gleichmäßig an
    void setDispersionRamp(float target, size_t ramp_samples) noexcept {
        float clamped = std::max(-0.95f, std::min(target, 0.95f));
        target_dispersion_ = clamped;
        target_dispersion_ports_.fill(clamped);
        target_dispersion_mix_ = 1.0f;
        dispersion_enabled_ = true;

        if (ramp_samples <= 1) {
            setDispersion(clamped, true);
            return;
        }

        dispersion_ramp_samples_ = ramp_samples;
        dispersion_ramp_counter_ = 0;
        dispersion_is_ramping_ = true;
        for (size_t i = 0; i < 4; ++i) {
            dispersion_ramp_step_[i] = (target_dispersion_ports_[i] - current_dispersion_ports_[i]) / static_cast<float>(ramp_samples);
        }
    }

    /// Sample-genaue lineare Host-Automation für einen einzelnen Port
    void setPortDispersionRamp(size_t port, float target, size_t ramp_samples) noexcept {
        if (port >= 4) return;
        float clamped = std::max(-0.95f, std::min(target, 0.95f));
        target_dispersion_ports_[port] = clamped;
        target_dispersion_ = target_dispersion_ports_[0];
        target_dispersion_mix_ = 1.0f;
        dispersion_enabled_ = true;

        if (ramp_samples <= 1) {
            setPortDispersion(port, clamped, true);
            return;
        }

        dispersion_ramp_samples_ = ramp_samples;
        dispersion_ramp_counter_ = 0;
        dispersion_is_ramping_ = true;
        dispersion_ramp_step_.fill(0.0f);
        dispersion_ramp_step_[port] = (target_dispersion_ports_[port] - current_dispersion_ports_[port]) / static_cast<float>(ramp_samples);
    }

    /// Prüft, ob aktuell noch eine Glättung oder Rampe aktiv ist (Host-Inspektor)
    bool isDispersionSmoothing() const noexcept {
        if (dispersion_is_ramping_) return true;
        if (dispersion_crossfade_active_) return true;
        if (nodes_.size() > 0 && nodes_[0].isDispersionCrossfading()) return true;
        if (std::abs(target_dispersion_mix_ - current_dispersion_mix_) > 1e-5f) return true;
        for (size_t i = 0; i < 4; ++i) {
            if (std::abs(target_dispersion_ports_[i] - current_dispersion_ports_[i]) > 1e-4f) return true;
        }
        return false;
    }

    float getTargetPortDispersion(size_t port) const noexcept {
        return (port < 4) ? target_dispersion_ports_[port] : 0.0f;
    }

    float getCurrentPortDispersion(size_t port) const noexcept {
        return (port < 4) ? current_dispersion_ports_[port] : 0.0f;
    }

    // Parameter Getters & Inspektoren
    float getRoomSize() const noexcept { return target_room_size_; }
    float getCurrentRoomSize() const noexcept { return current_room_size_; }
    float getRoomSizeSmoothingTime() const noexcept { return room_size_smooth_ms_; }
    float getChorusRate() const noexcept { return chorus_rate_hz_; }
    float getChorusDepth() const noexcept { return target_chorus_depth_; }
    float getCurrentChorusDepth() const noexcept { return current_chorus_depth_; }
    float getDecayTime() const noexcept { return t60_seconds_; }
    float getDampingCutoff() const noexcept { return damping_cutoff_hz_; }
    float getDryWet() const noexcept { return wet_fraction_; }

    // ========================================================================
    // Audio Processing
    // ========================================================================

    /// Verarbeitet ein einzelnes Stereo-Sampleframe (sample-genau, kausal, allokationsfrei)
    inline std::pair<float, float> processSample(float in_l, float in_r) noexcept {
        smoothParameters();
        lfo_bank_.step(current_chorus_depth_);

        if (++gain_update_counter_ >= kGainUpdateInterval) {
            gain_update_counter_ = 0;
            recalculateGainsAndDelays(false);
        }

        // Schritt 0: Orthogonale 16-Knoten Walsh-Hadamard Injektion
        // Vollständige Energieerhaltung: sum_u (inj_u)^2 = in_l^2 + in_r^2
        alignas(16) float node_injections[kNumNodes];
        const float* hl = kHadamard16[in_l_row_];
        const float* hr = kHadamard16[in_r_row_];
        for (size_t u = 0; u < kNumNodes; ++u) {
            node_injections[u] = in_l * hl[u] + in_r * hr[u];
        }

        alignas(16) Vec4 node_outputs[kNumNodes];

        // Schritt 1: 16 Hypercube-Knoten einlesen mit modulierten LFO-Delays
        for (size_t u = 0; u < kNumNodes; ++u) {
            alignas(16) std::array<float, 4> dynamic_delays;
            for (size_t k = 0; k < kNumPorts; ++k) {
                size_t edge = edge_map_[u][k];
                float mod = lfo_bank_.getModulation(edge);
                float d = nominal_delays_[u][k] * current_room_size_ + mod;
                dynamic_delays[k] = std::max(12.0f, d);
            }

            Vec4 in_ports = delay_bank_.readNodePorts(u * kNumPorts, dynamic_delays);

            // Injektion gleichmäßig auf die 4 Knoten-Ports (Faktor 0.5f normiert die Energie pro Knoten)
            Vec4 injection = SimdBackend::set1(node_injections[u] * 0.5f);

            node_outputs[u] = nodes_[u].process(in_ports, injection);
        }

        // Schritt 2: Shift-Permutation S(u, k) = (u ^ 2^k, k) und Delay-Bank schreiben
        for (size_t u = 0; u < kNumNodes; ++u) {
            alignas(16) std::array<float, 4> outs;
            SimdBackend::storeu(outs.data(), node_outputs[u]);

            for (size_t k = 0; k < kNumPorts; ++k) {
                size_t v = u ^ (1ULL << k);
                size_t target_line = v * kNumPorts + k;
                delay_bank_.write(target_line, outs[k]);
            }
        }

        // Schritt 3: Delay-Bank Pointer synchron inkrementieren
        delay_bank_.advance();

        // Schritt 4: Orthogonale 16-Knoten Hadamard-Auskopplung (Stereo)
        // An jedem Knoten greifen wir den antisymmetrischen Grover-Eigenraum [1, -1, 1, -1] ab.
        // Dieser steht orthogonal auf dem Injektions-Eigenraum [1, 1, 1, 1] (kein Direktdurchschlag bei t=0).
        alignas(16) float node_readouts[kNumNodes];
        for (size_t u = 0; u < kNumNodes; ++u) {
            alignas(16) std::array<float, 4> outs;
            SimdBackend::storeu(outs.data(), node_outputs[u]);
            node_readouts[u] = 0.5f * (outs[0] - outs[1] + outs[2] - outs[3]);
        }

        const float* out_hl = kHadamard16[out_l_row_];
        const float* out_hr = kHadamard16[out_r_row_];
        float wet_l = 0.0f;
        float wet_r = 0.0f;
        for (size_t u = 0; u < kNumNodes; ++u) {
            wet_l += out_hl[u] * node_readouts[u];
            wet_r += out_hr[u] * node_readouts[u];
        }

        float out_l = in_l * (1.0f - wet_fraction_) + wet_l * wet_fraction_;
        float out_r = in_r * (1.0f - wet_fraction_) + wet_r * wet_fraction_;

        return { out_l, out_r };
    }

    /// Block-Verarbeitung für DAW AudioBuffers
    void processBlock(float* const* channel_data, size_t num_channels, size_t num_samples) noexcept {
        float* out_l = channel_data[0];
        float* out_r = (num_channels > 1) ? channel_data[1] : channel_data[0];

        for (size_t i = 0; i < num_samples; ++i) {
            float in_l = out_l[i];
            float in_r = (num_channels > 1) ? out_r[i] : in_l;
            auto [rev_l, rev_r] = processSample(in_l, in_r);
            out_l[i] = rev_l;
            if (num_channels > 1) {
                out_r[i] = rev_r;
            }
        }
    }

private:
    void initEdgeTopology() noexcept {
        for (auto& row : edge_map_) row.fill(0);
        size_t edge_count = 0;
        for (size_t u = 0; u < kNumNodes; ++u) {
            for (size_t k = 0; k < kNumPorts; ++k) {
                size_t v = u ^ (1ULL << k);
                if (u < v) {
                    edge_map_[u][k] = edge_count;
                    edge_map_[v][k] = edge_count;
                    ++edge_count;
                }
            }
        }

        // 4 Grund-Delays für die 4 Dimensionen des Hypercubes (teilerfremd)
        const std::array<float, 4> base_delays = { 439.0f, 571.0f, 743.0f, 997.0f };
        for (size_t u = 0; u < kNumNodes; ++u) {
            for (size_t k = 0; k < kNumPorts; ++k) {
                size_t v = u ^ (1ULL << k);
                // Reziprokes Detuning (+-2.5 Samples) bricht Symmetrie ohne Kausalitätsbruch
                float detune = (u < v) ? 2.5f : -2.5f;
                nominal_delays_[u][k] = base_delays[k] + detune;
            }
        }
    }

    void updateSmoothingCoefficients() noexcept {
        const float fs = static_cast<float>(sample_rate_);
        float tau_room = (room_size_smooth_ms_ * 0.001f);
        room_size_alpha_ = 1.0f - std::exp(-1.0f / (tau_room * fs));

        float tau_depth = 0.050f; // 50ms Glättung für Chorus-Tiefe
        chorus_depth_alpha_ = 1.0f - std::exp(-1.0f / (tau_depth * fs));

        float tau_coin = (coin_phase_smooth_ms_ * 0.001f);
        coin_phase_alpha_ = 1.0f - std::exp(-1.0f / (tau_coin * fs));

        float tau_disp = (dispersion_smooth_ms_ * 0.001f);
        dispersion_alpha_ = 1.0f - std::exp(-1.0f / (tau_disp * fs));

        float tau_mix = 0.015f; // 15ms klickfreies Crossfade für Enable/Bypass
        dispersion_mix_alpha_ = 1.0f - std::exp(-1.0f / (tau_mix * fs));
    }

    void updateCoinCoefficients() noexcept {
        const float c = std::cos(current_coin_phase_);
        const float s = std::sin(current_coin_phase_);
        const float a = -0.5f * c;
        const float b =  0.5f * (1.0f - s);
        const float d =  0.5f * (1.0f + s);

        coin_coeff_a_    = SimdBackend::set1(a);
        coin_coeff_odd_  = SimdBackend::set(b, d, d, b);
        coin_coeff_even_ = SimdBackend::set(d, b, b, d);

        for (auto& node : nodes_) {
            node.setCoinCoefficients(coin_coeff_a_, coin_coeff_odd_, coin_coeff_even_);
        }
    }

    void updateDispersionCoefficients() noexcept {
        dispersion_coeff_ = SimdBackend::loadu(current_dispersion_ports_.data());
        for (auto& node : nodes_) {
            node.setDispersionCoefficients(dispersion_coeff_);
        }
    }

    inline void smoothParameters() noexcept {
        current_room_size_ += room_size_alpha_ * (target_room_size_ - current_room_size_);
        if (std::abs(target_room_size_ - current_room_size_) < 1e-5f) {
            current_room_size_ = target_room_size_;
        }

        current_chorus_depth_ += chorus_depth_alpha_ * (target_chorus_depth_ - current_chorus_depth_);
        if (std::abs(target_chorus_depth_ - current_chorus_depth_) < 1e-5f) {
            current_chorus_depth_ = target_chorus_depth_;
        }

        if (std::abs(target_coin_phase_ - current_coin_phase_) > 1e-6f) {
            current_coin_phase_ += coin_phase_alpha_ * (target_coin_phase_ - current_coin_phase_);
            if (std::abs(target_coin_phase_ - current_coin_phase_) < 1e-6f) {
                current_coin_phase_ = target_coin_phase_;
            }
            updateCoinCoefficients();
        }

        // Dispersion Mix Glättung (klickfreies Bypassen/Aktivieren)
        if (std::abs(target_dispersion_mix_ - current_dispersion_mix_) > 1e-5f) {
            current_dispersion_mix_ += dispersion_mix_alpha_ * (target_dispersion_mix_ - current_dispersion_mix_);
            if (std::abs(target_dispersion_mix_ - current_dispersion_mix_) < 1e-5f) {
                current_dispersion_mix_ = target_dispersion_mix_;
            }
            for (auto& node : nodes_) {
                node.setDispersionMix(current_dispersion_mix_);
            }
        }

        // Dispersion Koeffizienten-Glättung (sample-genau linear gerampt, Zustands-Crossfade oder exp-Glättung)
        bool disp_changed = false;
        if (dispersion_is_ramping_) {
            disp_changed = true;
            ++dispersion_ramp_counter_;
            for (size_t i = 0; i < 4; ++i) {
                current_dispersion_ports_[i] += dispersion_ramp_step_[i];
            }
            if (dispersion_ramp_counter_ >= dispersion_ramp_samples_) {
                current_dispersion_ports_ = target_dispersion_ports_;
                dispersion_is_ramping_ = false;
            }
            current_dispersion_ = current_dispersion_ports_[0];
            updateDispersionCoefficients();
        } else if (dispersion_crossfade_active_) {
            ++dispersion_crossfade_counter_;
            float t = static_cast<float>(dispersion_crossfade_counter_) / static_cast<float>(dispersion_crossfade_samples_);
            if (t >= 1.0f) {
                t = 1.0f;
                current_dispersion_ports_ = target_dispersion_ports_;
                current_dispersion_ = target_dispersion_;
                dispersion_crossfade_active_ = false;
            } else {
                float p = t * t * (3.0f - 2.0f * t);
                for (size_t i = 0; i < 4; ++i) {
                    current_dispersion_ports_[i] = dispersion_start_ports_[i] + p * (target_dispersion_ports_[i] - dispersion_start_ports_[i]);
                }
                current_dispersion_ = current_dispersion_ports_[0];
            }
        } else {
            bool disp_remaining = false;
            for (size_t i = 0; i < 4; ++i) {
                float diff = target_dispersion_ports_[i] - current_dispersion_ports_[i];
                if (std::abs(diff) > 1e-4f) {
                    current_dispersion_ports_[i] += dispersion_alpha_ * diff;
                    if (std::abs(target_dispersion_ports_[i] - current_dispersion_ports_[i]) < 1e-4f) {
                        current_dispersion_ports_[i] = target_dispersion_ports_[i];
                    } else {
                        disp_remaining = true;
                    }
                    disp_changed = true;
                } else {
                    current_dispersion_ports_[i] = target_dispersion_ports_[i];
                }
            }
            if (!disp_remaining) current_dispersion_ports_ = target_dispersion_ports_;
            current_dispersion_ = current_dispersion_ports_[0];

            if (disp_changed) {
                updateDispersionCoefficients();
            }
        }
    }

    void recalculateGainsAndDelays(bool force) noexcept {
        float delta_room = std::abs(current_room_size_ - last_gain_room_size_);
        if (!force && delta_room < 0.005f) {
            return;
        }
        last_gain_room_size_ = current_room_size_;

        const float fs = static_cast<float>(sample_rate_);
        for (size_t u = 0; u < kNumNodes; ++u) {
            std::array<float, 4> cutoffs;
            std::array<float, 4> gains;
            for (size_t k = 0; k < kNumPorts; ++k) {
                float effective_delay = nominal_delays_[u][k] * current_room_size_;
                float delay_time_s = effective_delay / fs;
                // Exakter T60 Loop-Gain: g = 10^(-3 * delay_s / T60)
                gains[k] = std::pow(10.0f, -3.0f * delay_time_s / t60_seconds_);
                cutoffs[k] = damping_cutoff_hz_;
            }
            nodes_[u].setDamping(cutoffs, gains, sample_rate_);
        }
    }

    double sample_rate_{48000.0};
    DTQWDelayLineBank<SimdBackend> delay_bank_;
    DTQWLfoBank32<SimdBackend> lfo_bank_;
    std::array<DTQWNode<SimdBackend, NumAllpassStages>, kNumNodes> nodes_;

    std::array<std::array<size_t, 4>, kNumNodes> edge_map_{};
    std::array<std::array<float, 4>, kNumNodes> nominal_delays_{};

    // Steuerbare Parameter & Glättung
    float target_room_size_{1.0f};
    float current_room_size_{1.0f};
    float room_size_smooth_ms_{80.0f};
    float room_size_alpha_{0.001f};

    float chorus_rate_hz_{0.8f};
    float target_chorus_depth_{3.0f};
    float current_chorus_depth_{3.0f};
    float chorus_depth_alpha_{0.001f};

    float target_coin_phase_{0.0f};
    float current_coin_phase_{0.0f};
    float coin_phase_smooth_ms_{20.0f};
    float coin_phase_alpha_{0.001f};
    Vec4 coin_coeff_a_{};
    Vec4 coin_coeff_odd_{};
    Vec4 coin_coeff_even_{};

    // Steuerbare Dispersion & geglättete Host-Automation
    float target_dispersion_{0.0f};
    float current_dispersion_{0.0f};
    alignas(16) std::array<float, 4> target_dispersion_ports_{0.0f, 0.0f, 0.0f, 0.0f};
    alignas(16) std::array<float, 4> current_dispersion_ports_{0.0f, 0.0f, 0.0f, 0.0f};
    alignas(16) std::array<float, 4> dispersion_ramp_step_{0.0f, 0.0f, 0.0f, 0.0f};
    size_t dispersion_ramp_samples_{0};
    size_t dispersion_ramp_counter_{0};
    bool dispersion_is_ramping_{false};

    float dispersion_smooth_ms_{20.0f};
    float dispersion_alpha_{0.001f};
    bool dispersion_enabled_{false};
    float target_dispersion_mix_{0.0f};
    float current_dispersion_mix_{0.0f};
    float dispersion_mix_alpha_{0.001f};
    Vec4 dispersion_coeff_{};

    // Zustandsübergreifender Crossfade beim Ändern der Allpass-Koeffizienten
    bool dispersion_crossfade_enabled_{true};
    float dispersion_crossfade_ms_{15.0f};
    size_t dispersion_crossfade_samples_{720};
    size_t dispersion_crossfade_counter_{0};
    bool dispersion_crossfade_active_{false};
    alignas(16) std::array<float, 4> dispersion_start_ports_{0.0f, 0.0f, 0.0f, 0.0f};

    float t60_seconds_{1.8f};
    float damping_cutoff_hz_{5500.0f};
    float wet_fraction_{1.0f};

    static constexpr size_t kGainUpdateInterval = 32;
    size_t gain_update_counter_{0};
    float last_gain_room_size_{1.0f};

    int in_l_row_{1};  // Standard: Walsh-Dipol Bit 0
    int in_r_row_{2};  // Standard: Walsh-Dipol Bit 1
    int out_l_row_{5}; // Standard: Walsh-Quadrupol Bit 0 + Bit 2
    int out_r_row_{6}; // Standard: Walsh-Quadrupol Bit 1 + Bit 2
};

/// Typ-Alias für Standard-4D-Hypercube DTQW Reverb mit 2 Allpass-Dispersionsstufen
template <typename SimdBackend = simd::DefaultBackend>
using DTQWReverb = DTQWEngine<SimdBackend, 2>;

} // namespace dtqw
