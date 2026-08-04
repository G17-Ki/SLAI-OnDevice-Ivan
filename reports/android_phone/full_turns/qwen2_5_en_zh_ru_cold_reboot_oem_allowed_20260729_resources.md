# Cold-reboot OEM-allowed resource snapshot — 2026-07-29

This snapshot records the verified phone-resident stack after the OEM
AutoLaunch allowance enabled the main app and runtime companion to recover from
a cold reboot. It covers the persistent speech-agent processes after startup.

## Memory

| Metric | Aggregate value |
|---|---:|
| Point-in-time PSS | 4,368,319 KiB |
| Point-in-time RSS from `dumpsys meminfo` | 4,846,284 KiB |
| Point-in-time RSS from `/proc` | 4,764,484 KiB |
| Summed per-process `VmHWM` | 4,921,676 KiB upper bound |

The two current RSS totals come from different Android/Linux accounting views
and sampling points, so a small difference is expected. `VmHWM` is each
process's lifetime high-water RSS. Its aggregate is a conservative upper bound,
not simultaneous current memory use or a measured combined peak.

## Thermal and stability state

- Android thermal status: 0.
- Current HAL maxima: CPU approximately 30.8 °C, GPU approximately 30.8 °C,
  and NSP approximately 30.5 °C.
- Skin: 27.62 °C; battery: 29 °C.
- All current cooling-device values: 0.
- Filtered evidence contained no matching Android crash, LMK/low-memory kill,
  freezer event, native-TTS failure, or QNN/HTP failure.

## OEM process state

- Main app: `isPendingFreeze=false`, `isFrozen=false`.
- Runtime companion: `isPendingFreeze=false`, `isFrozen=false`.
- Recorded child-process state: `killed=false`.

This capture therefore shows the complete resident stack running without an
active thermal, freezer, low-memory-kill, or accelerator-failure signal during
the verified post-reboot observation window.
