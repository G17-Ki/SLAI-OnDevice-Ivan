# Cold-reboot acceptance provenance — 2026-07-29

This note binds the cold-reboot EN/ZH/RU result to the exact APK pair installed
on the Nubia NX809J. It is a provenance supplement to the raw acceptance JSON,
whose schema records runtime/backend identities but not APK SHA-256 values.

## Device and boot identity

- ADB serial: redacted from the public report
- Build fingerprint:
  `REDMAGIC/NX809J/NX809J:16/BQ2A.250705.001-BP2A.250605.031.A3/20260701.145746:user/release-keys`
- Linux boot ID: `2ae1f997-1d0f-47c3-971f-b4883a677f1b`
- Kernel boot time (`/proc/stat` `btime`): `2026-07-29T13:45:18+0800`
- `sys.boot_completed`: `1`
- Read-only provenance inspection: `2026-07-29T14:22:33+0800`

The acceptance harness ran from `2026-07-29T13:55:52+0800` through
`2026-07-29T13:56:15+0800`, about ten minutes after this boot. Telemetry was
pulled at `2026-07-29T13:58:22+0800`.

## Installed artifacts

| Package | Version | Device `lastUpdateTime` | Device APK SHA-256 |
|---|---|---|---|
| `ai.liquid.lfmdemo` | code 26, `0.6.0-unplugged` | `2026-07-29 13:16:48` | `2d9a90ce90824d15f143f7847bcbce21bbb3141de3dd011eadeb24db0935871b` |
| `ai.liquid.lfmdemo.runtime` | code 1, `0.1.0-unplugged-runtime` | `2026-07-29 13:04:09` | `f2f5e805ca524d2fdde7c62ba2ef3dc1e48a9818f105e447347ac37c5bf2070d` |

Both package update times precede the cold boot. The device hashes match the
two local build artifacts exactly.

## Process continuity

The same four post-boot PIDs recorded before and after the acceptance run were
still alive during the provenance inspection:

| PID | UID | Process | Phone-resident payload |
|---:|---:|---|---|
| 10499 | 10538 | `ai.liquid.lfmdemo` | main application |
| 10598 | 10514 | `ai.liquid.lfmdemo.runtime` | companion foreground service |
| 10620 | 10538 | `liblfmserver.so` | LFM files under main-app private storage; loopback port 8080 |
| 10653 | 10514 | `libqwenserver.so` | Qwen2.5 GGUF under companion-private storage; loopback port 6020 |

An Android package replacement stops its application process. Because the APK
update times predate this boot and all four PIDs remained continuous from the
main-only cold start through the turn matrix, telemetry pull, host-ADB stop,
and this inspection, the recorded turns belong to the exact APK hashes above.

## Result references

- Raw full-turn evidence:
  `qwen2_5_en_zh_ru_cold_reboot_oem_allowed_20260729.json`
- Raw evidence SHA-256:
  `68721ac76d1dcd144a2a1fa136c15291d9c4deadff55e8bcc6c326a49b0a010a`
- Telemetry directory:
  `qwen2_5_en_zh_ru_cold_reboot_oem_allowed_20260729_telemetry/`
- Resource snapshot:
  `qwen2_5_en_zh_ru_cold_reboot_oem_allowed_20260729_resources.md`

This proves the exact current build passed one cold reboot, first unlock,
main-only launch, and the three-language matrix. It does not erase the latent
fallback state-machine risks or substitute for repeated cold-reboot soak and a
literal physical USB-cable removal.
