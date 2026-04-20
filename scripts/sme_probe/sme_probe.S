// sme_probe.S — AArch64 trampolines for the SME probe.
//
// Each trampoline executes exactly one SME instruction between two
// easily-identifiable boundaries, then returns. If the SME
// instruction faults, our VEH advances PC by 4 so the `ret` still
// runs and the trampoline returns cleanly to C.

    .text

    .globl  probe_smstart_sm
    .p2align 2
probe_smstart_sm:
    smstart sm                  // enter Streaming-SVE mode
    smstop  sm                  // leave (no-op if fault skipped smstart)
    ret

    .globl  probe_smstart_za
    .p2align 2
probe_smstart_za:
    smstart za                  // enable ZA tile storage
    smstop  za
    ret

    .globl  probe_zero_za
    .p2align 2
probe_zero_za:
    smstart                     // enable both sm and za
    zero    {za}                // ZA-tile op: clear all ZA tiles
    smstop
    ret
