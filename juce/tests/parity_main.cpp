#include "../Source/Dsp/DTQWReverb.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

int main (int argc, char** argv)
{
    std::printf("==> DTQW Reverb DSP Parity & Self-Test running...\n");

    dtqw::DTQWReverb<dtqw::simd::DefaultBackend> engine;
    engine.prepare (48000.0);
    engine.setRoomSize (1.0f, true);
    engine.setDecayTime (1.8f);
    engine.setDampingCutoff (5500.0f);
    engine.setCoinPhase (0.0f, true);
    engine.setChorusDepth (3.0f, true);
    engine.setChorusRate (0.8f);
    engine.setDryWet (1.0f);
    engine.setDispersionEnabled (true, true);
    engine.setDispersion (0.45f, true);
    engine.setHadamardRouting (1, 2, 5, 6);

    std::vector<float> L (128, 0.0f);
    std::vector<float> R (128, 0.0f);
    L[0] = 1.0f; R[0] = 1.0f; // Unit Dirac impulse

    float maxPeak = 0.0f;
    float sumEnergy = 0.0f;
    bool hasNanInf = false;

    float* channels[2] = { L.data(), R.data() };

    for (int block = 0; block < 400; ++block)
    {
        if (block > 0)
        {
            std::fill (L.begin(), L.end(), 0.0f);
            std::fill (R.begin(), R.end(), 0.0f);
        }

        // Stress parameter variations along the way
        if (block == 100) engine.setCoinPhase (1.570796f);
        if (block == 200) engine.setRoomSize (1.5f);
        if (block == 300) engine.setDispersion (-0.5f);

        engine.processBlock (channels, 2, 128);

        for (int i = 0; i < 128; ++i)
        {
            float sl = L[(size_t) i], sr = R[(size_t) i];
            if (std::isnan (sl) || std::isinf (sl) || std::isnan (sr) || std::isinf (sr))
                hasNanInf = true;

            float p = std::max (std::abs (sl), std::abs (sr));
            if (p > maxPeak) maxPeak = p;
            sumEnergy += (sl * sl + sr * sr);
        }
    }

    std::printf("   Peak: %.6f, Total Energy: %.6f, NaN/Inf: %s\n",
                maxPeak, sumEnergy, hasNanInf ? "DETECTED (FAIL)" : "NONE (OK)");

    if (hasNanInf || maxPeak <= 0.0f)
    {
        std::fprintf(stderr, "   ERROR: Self-test failed!\n");
        return 1;
    }

    std::printf("   DSP-Selbsttest: OK\n");
    return 0;
}
