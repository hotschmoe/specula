# Probe Windows IsProcessorFeaturePresent for ARM64 feature flags.
# See: https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
#
# Feature IDs from winnt.h that are relevant to ARM64 / SME investigation:
#   PF_ARM_VFP_32_REGISTERS_AVAILABLE        = 18
#   PF_ARM_NEON_INSTRUCTIONS_AVAILABLE       = 19
#   PF_ARM_V8_INSTRUCTIONS_AVAILABLE         = 29
#   PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE  = 30
#   PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE   = 31
#   PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE = 34
#   PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE     = 43
#   PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE  = 44
#   PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE  = 45
#   PF_ARM_SVE_INSTRUCTIONS_AVAILABLE        = 46
#   PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE       = 47  (Windows 11 24H2+)
#   PF_ARM_SVE2P1_INSTRUCTIONS_AVAILABLE     = 48
#   PF_ARM_SME_INSTRUCTIONS_AVAILABLE        = 51  (may not be defined on all SDKs)
#   PF_ARM_SME2_INSTRUCTIONS_AVAILABLE       = 52  (may not be defined on all SDKs)
# Higher numbers are speculative; we probe 0..70 to catch any unnamed flags.

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class NativeArm {
    [DllImport("kernel32.dll")]
    public static extern bool IsProcessorFeaturePresent(uint ProcessorFeature);
}
'@

$names = @{
    18 = 'VFP_32_REGISTERS'
    19 = 'NEON'
    29 = 'V8'
    30 = 'V8_CRYPTO'
    31 = 'V8_CRC32'
    34 = 'V81_ATOMIC'
    43 = 'V82_DP'
    44 = 'V83_JSCVT'
    45 = 'V83_LRCPC'
    46 = 'SVE'
    47 = 'SVE2'
    48 = 'SVE2P1'
    49 = 'SVE_BF16 (speculative)'
    50 = 'SVE_I8MM (speculative)'
    51 = 'SME (speculative)'
    52 = 'SME2 (speculative)'
    53 = 'SME_F64F64 (speculative)'
    54 = 'SME_I16I64 (speculative)'
}

"Probe: IsProcessorFeaturePresent(N) for N in 0..70"
for ($i = 0; $i -le 70; $i++) {
    $present = [NativeArm]::IsProcessorFeaturePresent([uint32]$i)
    if ($present) {
        $name = $names[$i]
        if (-not $name) { $name = '(unnamed)' }
        "  [$i] = TRUE  ($name)"
    }
}
""
"Specifically for the SME investigation:"
foreach ($id in 46,47,48,51,52) {
    $present = [NativeArm]::IsProcessorFeaturePresent([uint32]$id)
    "  PF[$id] ($($names[$id])) = $present"
}
