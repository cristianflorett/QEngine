// The host-side TP planning arithmetic (make_plan / plan_groups), lifted out of flash.cu unchanged:
// a CPU-only ctest target (tests/flash_tp5_plan_unit) links this one translation unit and checks the
// planning rules with no engine, no CUDA call and no GPU. flash_run and qflash compile this file as
// well, so the definitions stay exactly where flash.h declares them for everyone.
#include "flash.h"

namespace flash {
std::vector<Plan> make_plan(int ngpu, const std::vector<double>& wts) {
    // 1 rank: a pipeline stage (whole layers, make_stage_plans); TP: 2 to kMaxTp (5) ranks (the comm / exchange
    // buffers hold kCommMaxRanks ranks, k_idx_select gathers kMaxTp ranks' keys)
    if (ngpu < 1 || ngpu > kMaxTp || (int)wts.size() != ngpu)
        throw std::runtime_error("flash: 1 (pipeline stage), 2 to 5 tensor-parallel ranks and as many expert-group weights expected");
    static_assert(NQ % 6 == 0 && LNV % 6 == 0 && LNV % LNK == 0, "head splits over 2 and 3 cards");
    static_assert(NQ % 4 == 0 && LNV % 4 == 0 && (LNV / 4) % (LNV / LNK) == 0, "head splits over 4 cards (whole key heads per card)");
    static_assert(LNV % LNK == 0, "value heads per key head");
    // expert groups (e * 5 + g, e = 0..512 with 512 the shared expert) spread by a largest-deficit interleave in
    // proportion to the weights: consecutive groups of one expert land on different cards (per-token balance)
    const int ng = (NE + 1) * NGRP;
    double tot = 0;
    for (double w : wts) tot += w;
    std::vector<int> card(ng), cnt(ngpu, 0);
    for (int k = 0; k < ng; k++) {
        int best = 0;
        double bd = -1e30;
        for (int c = 0; c < ngpu; c++) {
            const double d = wts[c] / tot * (k + 1) - cnt[c];
            if (d > bd + 1e-12) { bd = d; best = c; }
        }
        card[k] = best;
        cnt[best]++;
    }
    std::vector<Plan> v(ngpu);
    for (int r = 0; r < ngpu; r++) {
        Plan& p = v[r];
        p.rank = r;
        p.ngpu = ngpu;
        p.gcard = card;
        // DeltaNet: value heads [v_begin, v_end) of this rank, the key heads they use (LNV / LNK value heads per key
        // head, shared at rank boundaries as at 3 cards); QSA: q heads [q_begin, q_end) of this rank (gate / o_proj /
        // combine). Uneven at 5 ranks (24 q heads -> 4, 5, 5, 5, 5; 48 value heads -> 9, 10, 9, 10, 10), so no range is
        // assumed equal to NQ / ngpu anywhere: q_capacity() sizes the exchange slots and the real ranges bound the loops.
        p.v0 = v_begin(r, ngpu);
        p.v1 = v_end(r, ngpu);
        p.k0 = k_begin(r, ngpu);
        p.k1 = k_end(r, ngpu);
        p.q0 = q_begin(r, ngpu);
        p.q1 = q_end(r, ngpu);
        if (p.q1 - p.q0 < 1 || p.q1 - p.q0 > q_capacity(ngpu))
            throw std::runtime_error("flash: empty QSA q head range for a tensor-parallel rank");
        if (p.v1 - p.v0 < 1 || p.k1 - p.k0 < 1)
            throw std::runtime_error("flash: empty DeltaNet head range for a tensor-parallel rank");
        if (p.k1 - p.k0 > LNK)
            throw std::runtime_error("flash: DeltaNet key head range larger than the key heads");
        const int g = VOCAB / 128, a = g * r / ngpu, b = g * (r + 1) / ngpu;
        p.voc0 = a * 128;
        p.voc1 = b * 128;
        p.period = 512 * ngpu;
        p.own0 = 512 * r;
        p.own = 512;
    }
    return v;
}

int plan_groups(const Plan& p) {
    int n = 0;
    for (int c : p.gcard) n += c == p.rank;
    return n;
}
}  // namespace flash
