// The CPU part of the TP5 fixture, shared by flash_tp5_unit.cu (the GPU run) and flash_tp5_plan_unit.cu
// (the ctest run: plan.cu alone, no CUDA call). The includer must have the planning helpers in scope
// (flash.h / pp_plan.h / comm.cuh) and define the require(ok, why) macro.
static void plan_checks() {
    flash_pp::PickIn pick{};
    pick.n=5;
    require(flash_pp::pick_par(pick).par==flash_pp::Par::PP,"automatic five-card policy must remain PP");
    flash_pp::Par explicit_mode;
    require(flash_pp::parse_par("tp",&explicit_mode) && explicit_mode==flash_pp::Par::TP,
            "explicit TP must not silently become PP");
    require(flash_pp::comm_wanted(5,explicit_mode,false,false),"explicit TP5 must allocate rank communication");
    for(int n=1;n<=5;++n) {
        const auto plans=make_plan(n,std::vector<double>(n,1.0));
        std::vector<int> qs(NQ),vs(LNV),voc(VOCAB),groups((NE+1)*NGRP);
        for(int r=0;r<n;++r) {
            const Plan& p=plans[r];
            require(p.q0==q_begin(r,n) && p.q1==q_end(r,n),"Q plan/helpers disagree");
            require(p.v0==v_begin(r,n) && p.v1==v_end(r,n),"V plan/helpers disagree");
            require(p.k0==k_begin(r,n) && p.k1==k_end(r,n),"K plan/helpers disagree");
            require(q_count(r,n)<=q_capacity(n),"head exceeds padded capacity");
            if(n<5) require(q_count(r,n)==NQ/n,"TP1-4 Q split changed");
            for(int h=p.q0;h<p.q1;++h) { ++qs[h]; require(q_owner(h,n)==r,"Q owner inverse"); }
            require(p.voc0>=0 && p.voc1<=VOCAB && p.voc0%128==0 && p.voc1%128==0,"unaligned/out-of-range vocab partition");
            for(int word=p.voc0;word<p.voc1;++word) ++voc[word];
            for(int h=p.v0;h<p.v1;++h) { ++vs[h]; require(h/3>=p.k0 && h/3<p.k1,"value maps outside local keys"); }
            for(size_t g=0;g<groups.size();++g) if(p.gcard[g]==r) ++groups[g];
            if(n==5) require(plan_groups(p)==513,"TP5 must assign 513 expert groups/rank");
        }
        for(int count:qs) require(count==1,"Q split gap/overlap");
        for(int count:vs) require(count==1,"V split gap/overlap");
        for(int count:groups) require(count==1,"expert split gap/overlap");
        for(int count:voc) require(count==1,"vocab split gap/overlap");
    }
    for(int n:{0,6}) { bool rejected=false; try { make_plan(n,std::vector<double>(n,1)); } catch(const std::runtime_error&) { rejected=true; } require(rejected,"invalid TP count accepted"); }
    unsigned used=0;
    for(int r=0;r<5;++r) for(int o=0;o<5;++o) {
        const unsigned bit=comm_pair_bit(r,o);
        require(bit<32,"pair bit index exceeds unsigned"); require(!(used&(1u<<bit)),"pair index collision"); used|=1u<<bit;
        if(r<4 && o<4) require(bit==unsigned(4*r+o),"legacy TP4 pair index changed");
    }
    require(used==((1u<<25)-1),"pair mask must cover 25 distinct bits");
    const int boundaries[]={4,9,14,19,23};
    const int owners[]={1,2,3,4,4};
    for(int i=0;i<5;++i) require(q_owner(boundaries[i],5)==owners[i],"TP5 boundary head ownership");
}
