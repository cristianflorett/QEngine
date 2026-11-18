// The CPU part of the TP5 fixture: the planning rules of tp5_plan_checks.h, linked against plan.cu
// alone. No CUDA call is made, so ctest passes this on a host without any GPU (the GPU binary
// flash_tp5_unit stays out of ctest and runs in a GPU window).
#include "flash/flash.h"
#include "flash/pp_plan.h"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#define require(ok, why) do { if (!(ok)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,why); throw std::runtime_error(why); } } while (0)
using namespace flash;
#include "tp5_plan_checks.h"
int main() {
    try {
        plan_checks();
        std::printf("flash_tp5_plan_unit: planning rules passed with no CUDA call\n");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"flash_tp5_plan_unit: %s\n",e.what()); return 1; }
}
