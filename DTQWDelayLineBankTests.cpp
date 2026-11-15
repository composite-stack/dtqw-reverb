// Standalone conformance tests for the DTQW delay-bank guard-zone and SIMD layout.
// Standard Build:
//   g++ -std=c++17 -O3 -Wall -Wextra -Werror DTQWDelayLineBankTests.cpp -o dtqw_tests
// ASan + UBSan Build:
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Werror DTQWDelayLineBankTests.cpp -o dtqw_tests_sanitized
//   ASAN_OPTIONS=detect_leaks=1:check_initialization_order=1:strict_init_order=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ./dtqw_tests_sanitized
// On x86, the SSE backend is tested when SSE is enabled by the compiler.
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#if defined(__SSE__)
  #include <xmmintrin.h>
#endif

namespace dtqw::simd {
struct ScalarBackend {
    using Vec4 = std::array<float, 4>;
    static Vec4 loadu(const float* p) noexcept { return {p[0], p[1], p[2], p[3]}; }
    static void storeu(float* p, Vec4 v) noexcept { for (int i=0;i<4;++i) p[i]=v[i]; }
    static void transpose4x4(Vec4& a, Vec4& b, Vec4& c, Vec4& d) noexcept {
        const Vec4 x{a[0],b[0],c[0],d[0]}, y{a[1],b[1],c[1],d[1]};
        const Vec4 z{a[2],b[2],c[2],d[2]}, w{a[3],b[3],c[3],d[3]};
        a=x; b=y; c=z; d=w;
    }
    static Vec4 add(Vec4 a, Vec4 b) noexcept { for(int i=0;i<4;++i)a[i]+=b[i]; return a; }
    static Vec4 sub(Vec4 a, Vec4 b) noexcept { for(int i=0;i<4;++i)a[i]-=b[i]; return a; }
    static Vec4 mul(Vec4 a, Vec4 b) noexcept { for(int i=0;i<4;++i)a[i]*=b[i]; return a; }
    static Vec4 fma(Vec4 a, Vec4 b, Vec4 c) noexcept { for(int i=0;i<4;++i)c[i]=std::fma(a[i],b[i],c[i]); return c; }
    static Vec4 hermite(Vec4 ym1, Vec4 y0, Vec4 y1, Vec4 y2, Vec4 f) noexcept {
        const Vec4 h{.5f,.5f,.5f,.5f}, o{1.5f,1.5f,1.5f,1.5f};
        const Vec4 two{2,2,2,2}, twohalf{2.5f,2.5f,2.5f,2.5f};
        const Vec4 c1=mul(h,sub(y1,ym1));
        const Vec4 c2=sub(add(ym1,mul(two,y1)),add(mul(twohalf,y0),mul(h,y2)));
        const Vec4 c3=add(mul(h,sub(y2,ym1)),mul(o,sub(y0,y1)));
        return fma(fma(fma(c3,f,c2),f,c1),f,y0);
    }
};

#if defined(__SSE__)
struct SSEBackend {
    using Vec4 = __m128;
    static Vec4 loadu(const float* p) noexcept { return _mm_loadu_ps(p); }
    static void storeu(float* p, Vec4 v) noexcept { _mm_storeu_ps(p,v); }
    static void transpose4x4(Vec4& a, Vec4& b, Vec4& c, Vec4& d) noexcept { _MM_TRANSPOSE4_PS(a,b,c,d); }
    static Vec4 add(Vec4 a, Vec4 b) noexcept { return _mm_add_ps(a,b); }
    static Vec4 sub(Vec4 a, Vec4 b) noexcept { return _mm_sub_ps(a,b); }
    static Vec4 mul(Vec4 a, Vec4 b) noexcept { return _mm_mul_ps(a,b); }
    static Vec4 fma(Vec4 a, Vec4 b, Vec4 c) noexcept {
      #if defined(__FMA__)
        return _mm_fmadd_ps(a,b,c);
      #else
        return _mm_add_ps(_mm_mul_ps(a,b),c);
      #endif
    }
    static Vec4 hermite(Vec4 ym1, Vec4 y0, Vec4 y1, Vec4 y2, Vec4 f) noexcept {
        const Vec4 h=_mm_set1_ps(.5f), o=_mm_set1_ps(1.5f);
        const Vec4 two=_mm_set1_ps(2.0f), twohalf=_mm_set1_ps(2.5f);
        const Vec4 c1=mul(h,sub(y1,ym1));
        const Vec4 c2=sub(add(ym1,mul(two,y1)),add(mul(twohalf,y0),mul(h,y2)));
        const Vec4 c3=add(mul(h,sub(y2,ym1)),mul(o,sub(y0,y1)));
        return fma(fma(fma(c3,f,c2),f,c1),f,y0);
    }
};
#endif
} // namespace dtqw::simd

namespace dtqw {
template <class Backend, std::size_t NumLines=64, std::size_t BufferOrder=4>
class DTQWDelayLineBank {
public:
    using Vec4=typename Backend::Vec4;
    static constexpr std::size_t kBufferSize=std::size_t{1}<<BufferOrder;
    static constexpr std::size_t kBufferMask=kBufferSize-1;
    static constexpr std::size_t kGuardSamples=4;
    static constexpr std::size_t kStride=kBufferSize+kGuardSamples;

    DTQWDelayLineBank(): storage_(NumLines*kStride,0.0f) {}
    void write(std::size_t line,float x) noexcept {
        float* b=storage_.data()+line*kStride;
        b[writePos_]=x;
        if(writePos_<kGuardSamples) b[kBufferSize+writePos_]=x;
    }
    void advance() noexcept { writePos_=(writePos_+1)&kBufferMask; }
    std::size_t writePosition() const noexcept { return writePos_; }
    float sample(std::size_t line,std::size_t index) const noexcept { return storage_[line*kStride+index]; }

    Vec4 readNodePorts(std::size_t firstLine,const std::array<float,4>& delay) const noexcept {
        std::array<float,4> frac{};
        std::array<std::size_t,4> base{};
        for(std::size_t p=0;p<4;++p) {
            const float whole=std::floor(delay[p]);
            frac[p]=delay[p]-whole;
            base[p]=(writePos_-static_cast<std::size_t>(whole)-2)&kBufferMask;
        }
        const float* b0=storage_.data()+(firstLine+0)*kStride;
        const float* b1=storage_.data()+(firstLine+1)*kStride;
        const float* b2=storage_.data()+(firstLine+2)*kStride;
        const float* b3=storage_.data()+(firstLine+3)*kStride;
        Vec4 r0=Backend::loadu(b0+base[0]),r1=Backend::loadu(b1+base[1]);
        Vec4 r2=Backend::loadu(b2+base[2]),r3=Backend::loadu(b3+base[3]);
        Backend::transpose4x4(r0,r1,r2,r3);
        return Backend::hermite(r3,r2,r1,r0,Backend::loadu(frac.data()));
    }
private:
    std::vector<float> storage_;
    std::size_t writePos_{0};
};
} // namespace dtqw

static void check(bool ok,const char* label) {
    if(!ok) { std::cerr<<"FAIL: "<<label<<'\n'; std::exit(1); }
}

template<class Backend>
static void testGuardAndTranspose(const char* label) {
    using Bank=dtqw::DTQWDelayLineBank<Backend,8,4>; // 16 ring samples + 4 guards
    Bank bank;
    // Write 32 samples so the ring wraps; the last four are mirrored at the guard.
    for(int n=0;n<32;++n) {
        for(int line=0;line<4;++line) bank.write(line,100.0f*line+float(n));
        bank.advance();
    }
    check(bank.writePosition()==0,"write pointer wraps at ring size");

    // In the real process order, the current sample is written before delayed
    // reads. Verify mirrored values for the first four positions after wrap.
    for(int i=0;i<4;++i) {
        for(int line=0;line<4;++line) bank.write(line,100.0f*line+float(32+i));
        check(bank.sample(0,16+i)==32.0f+i,"guard mirrors wrapped samples");
        if(i<3) bank.advance();
    }
    // Position is now four; return to zero only for the synthetic read setup.
    for(int i=0;i<13;++i) bank.advance();

    // Each lane uses a different fractional delay; the loads straddle the
    // ring boundary (base index 13, taps 13,14,15,guard[0]).
    const std::array<float,4> delays{1.25f,1.50f,1.75f,1.125f};
    auto vec=bank.readNodePorts(0,delays);
    std::array<float,4> actual{};
    Backend::storeu(actual.data(),vec);
    for(int p=0;p<4;++p) {
        const float expected=100.0f*p+(32.0f-delays[p]);
        check(std::abs(actual[p]-expected)<2.0e-4f,"transposition keeps each port lane aligned");
    }
    std::cout<<"PASS guard-zone crossing + SIMD transpose: "<<label<<'\n';
}

static void testSharedAdvance() {
    using Bank=dtqw::DTQWDelayLineBank<dtqw::simd::ScalarBackend,64,4>;
    Bank bank;
    for(std::size_t line=0;line<64;++line) bank.write(line,float(line));
    check(bank.writePosition()==0,"writes share but do not advance clock");
    bank.advance();
    check(bank.writePosition()==1,"one bank tick advances all lines together");
    for(std::size_t line=0;line<64;++line) bank.write(line,float(line+100));
    bank.advance();
    check(bank.writePosition()==2,"all lines use the same next write index");
    for(int n=2;n<16;++n) bank.advance();
    check(bank.writePosition()==0,"shared pointer wraps synchronously");
    std::cout<<"PASS shared write-pointer advance/wrap\n";
}

int main() {
    testGuardAndTranspose<dtqw::simd::ScalarBackend>("Scalar");
#if defined(__SSE__)
    testGuardAndTranspose<dtqw::simd::SSEBackend>("SSE");
#endif
    testSharedAdvance();
    std::cout<<"All DTQWDelayLineBank tests passed.\n";
}
