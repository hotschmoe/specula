// sme_probe.c — minimal Windows ARM64 probe to determine whether the
// current CPU+OS combination can actually execute SME instructions
// (smstart/smstop and a ZA tile op) without faulting.
//
// Background: llama.cpp's KleidiAI integration assumes SME works on
// any non-Linux, non-Apple ARM64 platform that was compiled with
// __ARM_FEATURE_SME. On Oryon v2 / Windows 11 build 28000 that
// assumption triggers STATUS_ILLEGAL_INSTRUCTION. We want concrete
// evidence: is the hardware missing SME, or has the OS not enabled
// user-mode ZA-tile state?
//
// Probes are kept in separate trampoline functions (see sme_probe.S)
// so the fault site is deterministic and we can map it back to a
// specific instruction rather than dealing with inline-asm placement.
//
// Exit codes:
//   0  — all probes passed (SME is fully usable)
//   1  — smstart sm (streaming-SVE state) faulted
//   2  — smstart za (ZA tile state) faulted
//   3  — ZA tile instruction (zero za) faulted
//   4  — unexpected fault
//
// The fault classification is informational — any non-zero exit means
// we cannot safely dispatch SME kernels on this machine.

#include <windows.h>
#include <stdio.h>

extern void probe_smstart_sm(void);
extern void probe_smstart_za(void);
extern void probe_zero_za(void);

typedef enum {
    PROBE_NONE,
    PROBE_SMSTART_SM,
    PROBE_SMSTART_ZA,
    PROBE_ZERO_ZA,
} probe_id;

static volatile probe_id g_current = PROBE_NONE;
static volatile LONG     g_faulted = 0;
static volatile DWORD    g_code    = 0;
static void *            g_fault_pc = NULL;

static LONG WINAPI veh(EXCEPTION_POINTERS *ep) {
    g_faulted = 1;
    g_code    = ep->ExceptionRecord->ExceptionCode;
    g_fault_pc = ep->ExceptionRecord->ExceptionAddress;
    // Skip past the offending instruction (4 bytes on AArch64) so we
    // return from the handler into the next instruction in the
    // trampoline — which is `ret`, letting the probe function return
    // normally even after a fault.
    ep->ContextRecord->Pc += 4;
    return EXCEPTION_CONTINUE_EXECUTION;
}

static int run_probe(probe_id which, void (*fn)(void), const char *label) {
    g_current = which;
    g_faulted = 0;
    g_code = 0;
    g_fault_pc = NULL;

    printf("probe: %-20s ", label);
    fflush(stdout);
    fn();
    if (g_faulted) {
        printf("FAULT  code=0x%08lX  pc=%p\n", g_code, g_fault_pc);
        return 1;
    } else {
        printf("OK\n");
        return 0;
    }
}

int main(void) {
    // Also re-query IsProcessorFeaturePresent for the same flags we
    // probe-executed; a mismatch between OS claim and runtime fault
    // is itself informative.
    printf("Windows build reports:\n");
    printf("  PF_ARM_V82_DP       (43) = %d\n", IsProcessorFeaturePresent(43));
    printf("  PF_ARM_SVE          (46) = %d\n", IsProcessorFeaturePresent(46));
    printf("  PF_ARM_SVE2         (47) = %d\n", IsProcessorFeaturePresent(47));
    printf("  PF_ARM_(speculative)(51) = %d\n", IsProcessorFeaturePresent(51));
    printf("  PF_ARM_(speculative)(52) = %d\n", IsProcessorFeaturePresent(52));
    printf("\n");

    AddVectoredExceptionHandler(1 /* first */, veh);

    int f_sm = run_probe(PROBE_SMSTART_SM, probe_smstart_sm, "smstart sm");
    int f_za = run_probe(PROBE_SMSTART_ZA, probe_smstart_za, "smstart za");
    int f_zero = run_probe(PROBE_ZERO_ZA, probe_zero_za, "zero za (ZA tile op)");

    printf("\nsummary: smstart-sm=%s  smstart-za=%s  zero-za=%s\n",
           f_sm ? "FAULT" : "OK",
           f_za ? "FAULT" : "OK",
           f_zero ? "FAULT" : "OK");

    if (f_sm)      return 1;
    if (f_za)      return 2;
    if (f_zero)    return 3;
    return 0;
}
