// CPU-only unit test of the PP x TP (pptp) parallelism planner at a TP2 x PP3 boot: 3 stages of 2 cards = 6 ranks,
// FLASH_PAR=pptp, FLASH_PP_MAP=0,1|2,3|4,5, FLASH_PP_LAYERS=16,16,16.
// Exercises the exact host functions the engine runs at boot (src/flash/flash_main.cpp / flash_pp.cpp):
//   parse_par("pptp") -> pptp_groups = flash_pp::parse_stage_map (FLASH_PP_MAP branch and the FLASH_PP_TP fallback)
//   -> pptp_bounds = flash_pp::parse_layers (the FLASH_PP_LAYERS override) or group_speeds + partition_cards
//   (pptp_bounds without the env override, equal cards, QE_P2P=1). pp_plan.h is host logic with no CUDA and no device
// headers (its own contract), so this unit needs no GPU and no env; the invalid configs must be REJECTED with a
// clear message, the valid ones must build the 3-stage / 6-rank plan with the right layers per stage.
#include "flash/pp_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define EXPECT(c)                                                                                         \
    do {                                                                                                  \
        if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; }               \
    } while (0)

static bool near_(double a, double b) { return std::fabs(a - b) < 1e-9; }
static bool bounds_ok(const std::vector<int>& b, int n) {
    if ((int)b.size() != n + 1 || b.front() != 0 || b.back() != flash_pp::kLayers) return false;
    for (int s = 0; s < n; s++)
        if (b[s + 1] <= b[s]) return false;  // an empty stage or unordered bounds
    return true;
}

using namespace flash_pp;

int main() {
    std::string err;

    // ---- FLASH_PAR=pptp selects the mode (parse_par, the boot's first planner call; auto_default as the server)
    {
        Par par = Par::TP;
        EXPECT(parse_par("pptp", &par, &err, /*auto_default=*/true));
        EXPECT(par == Par::PPTP);
        EXPECT(strcmp(par_name(par), "pptp") == 0);
        Par bad = Par::TP;
        err.clear();
        EXPECT(!parse_par("pptp3", &bad, &err));  // an unknown value is refused, not defaulted
        EXPECT(!err.empty());
    }

    // ---- the stage groups: FLASH_PP_MAP=0,1|2,3|4,5 over six devices (pptp_groups' map branch)
    std::vector<std::vector<int>> groups;
    {
        const std::vector<int> devs{0, 1, 2, 3, 4, 5};
        err.clear();
        groups = parse_stage_map("0,1|2,3|4,5", devs, 2, &err);
        EXPECT(err.empty());
        EXPECT(groups.size() == 3);  // 3 PP stages
        int ranks = 0;
        for (const auto& g : groups) {
            EXPECT(g.size() == 2);  // TP2 inside every stage
            ranks += (int)g.size();
        }
        EXPECT(ranks == 6);  // 6 workers on the boot's rank tables
        EXPECT(stage_map_str(groups) == "0,1|2,3|4,5");  // the boot banner's own round-trip
        EXPECT(ranks <= kMaxWorkers && 2 <= kMaxGroupTp);  // inside the engine's rank / group-size limits

        // the other boot path: FLASH_PP_TP=2 over the device list (no map) builds the very same plan
        err.clear();
        EXPECT(parse_stage_map(nullptr, devs, 2, &err) == groups && err.empty());
        std::printf("pptp plan: %s -> %zu stages, %d ranks\n", stage_map_str(groups).c_str(), groups.size(), ranks);
    }

    // ---- group rules the planner must refuse
    {
        const std::vector<int> devs{0, 1, 2, 3, 4, 5};
        err.clear();
        EXPECT(parse_stage_map("0,1|2,2", devs, 2, &err).empty());
        EXPECT(err.find("twice in one stage") != std::string::npos);
        err.clear();
        EXPECT(parse_stage_map("0,1,2,3|4,5", devs, 2, &err).empty());  // a 4-card group: TP4 is a separate item
        EXPECT(err.find("TP groups hold 1 to 3 cards") != std::string::npos);
        err.clear();
        EXPECT(parse_stage_map("0,1", devs, 2, &err).empty());
        EXPECT(err.find("at least 2 stages") != std::string::npos);
        // single cards may mix into the TP groups (the documented layout): 0,1 | 2 stays legal
        err.clear();
        const std::vector<std::vector<int>> mix = parse_stage_map("0,1|2", devs, 2, &err);
        EXPECT(!mix.empty() && err.empty() && mix.size() == 2 && mix[1].size() == 1);
    }

    // ---- FLASH_PP_LAYERS=16,16,16 over the 3 stages (parse_layers, pptp_bounds' env branch)
    {
        err.clear();
        const std::vector<int> b = parse_layers("16,16,16", 3, &err);
        EXPECT(err.empty());
        EXPECT(bounds_ok(b, 3));
        EXPECT(b[1] == 16 && b[2] == 32 && b[3] == 48);
        for (int s = 0; s < 3; s++) EXPECT(b[s + 1] - b[s] == 16);  // 16 layers per stage
        std::printf("FLASH_PP_LAYERS=16,16,16 -> bounds 0|16|32|48, 16 layers per stage\n");
    }
    // an uneven split is legal while the counts sum to kLayers
    {
        err.clear();
        const std::vector<int> b = parse_layers("15,16,17", 3, &err);
        EXPECT(err.empty() && bounds_ok(b, 3) && b[1] == 15 && b[2] == 31);
    }
    // rejections: the layers must sum to 48, as many counts as stages, every count >= 1
    {
        err.clear();
        EXPECT(parse_layers("16,16,15", 3, &err).empty());
        EXPECT(err.find("summing to 48") != std::string::npos);
        std::printf("reject (47 != 48 layers): %s\n", err.c_str());
        err.clear();
        EXPECT(parse_layers("16,16", 3, &err).empty());
        EXPECT(err.find("3 stages need 3 counts") != std::string::npos);
        err.clear();
        EXPECT(parse_layers("16,0,32", 3, &err).empty());
        EXPECT(err.find("positive layer counts") != std::string::npos);
    }

    // ---- the ground truth the KV=ram-under-stages sizing builds on: the layers holding per-token K / V are the
    // QSA ones (is_qsa, the li%4==3 rule) - 12 of the 48 - and the MTP block on the last stage adds the 13th.
    // stage_nq() (next commit) counts these per stage; the host-RAM math keeps its machine total at 13.
    {
        int qsa = 0;
        for (int li = 0; li < kLayers; li++)
            if (is_qsa(li)) qsa++;
        EXPECT(qsa == 12);
        EXPECT(is_qsa(3) && is_qsa(47) && !is_qsa(0) && !is_qsa(16));  // the rule li%4==3, ends on 47
        std::printf("KV layers: 12 QSA in [0,%d), +1 MTP on the last stage = 13 per machine\n", kLayers);
    }

    // ---- the auto split of the same boot without FLASH_PP_LAYERS: pptp_bounds' CPU core (equal cards, QE_P2P=1)
    {
        const std::vector<CardSpeed> gs = group_speeds(groups, {}, /*p2p=*/true);
        EXPECT(gs.size() == 3);
        for (const CardSpeed& c : gs) {
            EXPECT(near_(c.pf, 1.78) && near_(c.dc, 1.43));  // the TP2 speed-ups the fits give (pp_plan.h)
            EXPECT(near_(c.free, 2 * 22e9));                 // the stage's memory: its two cards
        }
        const std::vector<int> b = partition_cards(3, CostModel{}, 4096.0, DecodeCost{}, 0.5, gs, 22e9);
        EXPECT(bounds_ok(b, 3));
        int mn = kLayers, mx = 0;
        for (int s = 0; s < 3; s++) {
            const int w = b[s + 1] - b[s];
            EXPECT(w >= 1);
            mn = std::min(mn, w);
            mx = std::max(mx, w);
        }
        EXPECT(mn >= 10 && mx <= 26);  // equal-speed groups: a near-even split (the lm_head only trims the last)
        std::printf("auto split (no FLASH_PP_LAYERS): %d|%d|%d layers per stage\n", b[1] - b[0], b[2] - b[1], b[3] - b[2]);

        // ---- the same auto split as the engine computes it for an INT8 checkpoint (71c8e65: the group size comes from
        // the checkpoint's expert layout; pptp_bounds passes weight_group_bytes(format) = 998,400 B, the case this
        // deployment boots) and the worst-nq clamp read over those bounds, exactly as flash_capi computes it before
        // the Workers start. Whatever the split: the pipeline holds the machine's 13 KV layers, the clamp covers
        // every stage, and a heavy FIRST stage (24 layers: 6 QSA layers > the last stage's 4 + MTP) moves the worst
        // away from the MTP stage - the clamp follows the max, it does not assume the last stage is the heaviest.
        const std::vector<int> bi = partition_cards(3, CostModel{}, 4096.0, DecodeCost{}, 0.5, gs, 22e9, 998400.0);
        EXPECT(bounds_ok(bi, 3));
        int worst = 0, sum = 0;
        for (int s = 0; s < 3; s++) {
            const int q = stage_nq(s, 3, bi[s], bi[s + 1]);
            worst = std::max(worst, q);
            sum += q;
        }
        EXPECT(sum == kLayers / kQsaEvery + 1);
        for (int s = 0; s < 3; s++) EXPECT(worst >= stage_nq(s, 3, bi[s], bi[s + 1]));
        EXPECT(worst >= 5 && worst <= 7);  // 5 on a near-even split; a stage may hold up to 26 layers (6-7 QSA)
        EXPECT(stage_nq(0, 3, 0, 24) == 6 && stage_nq(2, 3, 41, flash_pp::kLayers) == 3);  // the stage-0-worst case:
        // a 24-layer first stage holds 6 QSA layers; the last stage's [41,48) holds 43, 47 + MTP = 3
        std::printf("INT8 auto split: %d|%d|%d layers, worst nq %d (sum %d)\n", bi[1] - bi[0], bi[2] - bi[1],
                    bi[3] - bi[2], worst, sum);
    }

    // ---- mode B interleave over the plan: c = 6 sequences circulate over n = 3 stages; the group count is the
    // engine's min(c, n + 1) (the extra group fills the pipeline bubble), cycling 0..ng-1
    {
        EXPECT(decode_ngroups(6, 3) == 4);
        const std::vector<int> g = decode_groups(6, 3);
        EXPECT(g.size() == 6);
        EXPECT(g[0] == 0 && g[1] == 1 && g[2] == 2 && g[3] == 3 && g[4] == 0 && g[5] == 1);
        EXPECT(decode_ngroups(3, 3) == 3);  // c = n: every stage one group
    }

    // ---- stage_nq: KV layers per stage for the FLASH_KV=ram pool under a pipeline (ctx > 262144): the QSA layers
    // of the stage's range (3, 7, ..., 47) + the MTP block's layer, which loads on the last stage only. The even
    // 16|16|16 split and its +-1 / +-2 shifts that keep every bound inside a DeltaNet gap (17|16|15, 18|15|15 -
    // the engine's uneven auto partitions): every one gives 4|4|5 and sums to the machine's 13.
    {
        const int splits[3][3] = {{16, 16, 16}, {17, 16, 15}, {18, 15, 15}};
        for (const auto& sp : splits) {
            const int b[4] = {0, sp[0], sp[0] + sp[1], flash_pp::kLayers};
            EXPECT(b[3] == 48);
            EXPECT(stage_nq(0, 3, b[0], b[1]) == 4);
            EXPECT(stage_nq(1, 3, b[1], b[2]) == 4);
            EXPECT(stage_nq(2, 3, b[2], b[3]) == 5);  // the MTP layer's pool rows live on the last stage
            EXPECT(stage_nq(0, 3, b[0], b[1]) + stage_nq(1, 3, b[1], b[2]) + stage_nq(2, 3, b[2], b[3]) ==
                   flash_pp::kLayers / flash_pp::kQsaEvery + 1);  // the same 13 a TP card holds
        }
        EXPECT(stage_nq(3, 3, 0, 48) == 13);   // a stage index out of the pipeline: the TP fallback 13
        EXPECT(stage_nq(-1, 3, 0, 48) == 13);  // (mode A's TP decode ranks size this way too)
        EXPECT(stage_nq(0, 1, 0, 48) == 13);   // no pipeline: the 12 QSA + MTP on the single card
        // pool bytes per token: one KV layer = kv_layer_bytes() = 1,040 B fp8 for the whole machine (flash.h); a TP2
        // stage rank holds half its stage's: 4 x 520 B/tok on stages 0-1, 5 x 520 on the last; both ranks of a stage
        // together: nq x 1,040 B/tok (the host-RAM plan of flash_capi keeps asking the 13-layer machine total)
        EXPECT(near_(stage_nq(0, 3, 0, 16) * kv_layer_bytes() / 2, 4 * 520.0));
        EXPECT(near_(stage_nq(2, 3, 32, 48) * kv_layer_bytes() / 2, 5 * 520.0));
        EXPECT(near_(stage_nq(2, 3, 32, 48) * kv_layer_bytes(), 5 * 1040.0));
    }

    // ---- stage_nq on TWO stages (PP2xTP2, the four-card experiment) and over EVERY partition of the layer line:
    // 24|24 splits the QSA layers 6|6 and the MTP layer tips the last stage to 7; and however the two cuts of a
    // three-stage split fall (every pair 1..47, degenerate ranges included) the whole pipeline still holds the 13
    // KV layers a TP card holds - the clamp's worst stage and the machine total read this same arithmetic.
    {
        EXPECT(stage_nq(0, 2, 0, 24) == 6);
        EXPECT(stage_nq(1, 2, 24, flash_pp::kLayers) == 7);  // 6 QSA + the MTP layer
        for (int a = 1; a < flash_pp::kLayers - 1; a++)
            for (int b = a + 1; b < flash_pp::kLayers; b++)
                EXPECT(stage_nq(0, 3, 0, a) + stage_nq(1, 3, a, b) + stage_nq(2, 3, b, flash_pp::kLayers) ==
                       flash_pp::kLayers / flash_pp::kQsaEvery + 1);
    }

    // ---- the pool-uniformity invariant (review F1 / F2): the pool ROWS are ONE object shared by all six ranks -
    // admission is the min of the ranks' kv_ram_periods and the file anchor reads W[0] - so two ranks of the same
    // stage MUST see the same cap_tok, and under the worst-nq clamp so MUST the six. alloc_state charges every rank
    // the plan's worst stage_nq, which makes the per-position bytes stage-independent: one 2,600 B/tok TP2 charge
    // (5 x 1,040 / 2) pays for identical rows everywhere, while what a card really pins stays its own: 2,080 B/tok
    // on the 4-layer stages, 2,600 on the last. The clamp covers every stage because worst >= every own nq.
    {
        const int b[4] = {0, 16, 32, flash_pp::kLayers};  // the even TP2xPP3 split (and any other gives the same)
        int worst = 0;
        for (int s = 0; s < 3; s++) worst = std::max(worst, stage_nq(s, 3, b[s], b[s + 1]));
        EXPECT(worst == 5);
        for (int s = 0; s < 3; s++)
            EXPECT(worst >= stage_nq(s, 3, b[s], b[s + 1]));  // identical rows fit the heaviest stage's array
        EXPECT(near_(worst * kv_layer_bytes() / 2, 2600.0));  // the one charge every rank's cap is computed from
        EXPECT(near_(stage_nq(0, 3, b[0], b[1]) * kv_layer_bytes() / 2, 2080.0));  // stage 0 pins less than charged
        EXPECT(near_(stage_nq(2, 3, b[2], b[3]) * kv_layer_bytes() / 2, 2600.0));  // the last stage pins all of it
    }

    if (fails) {
        fprintf(stderr, "QFN-PP-PLAN-UNIT: FAIL (%d checks failed)\n", fails);
        return 1;
    }
    std::printf("QFN-PP-PLAN-UNIT: PASS\n");
    return 0;
}
