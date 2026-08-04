# QAIRT / HTP v81 toolchain gate

Audit date: 2026-07-17 (Asia/Shanghai)  
Scope: read-only host, Qualcomm AI Hub, published package, connected-phone, and Qualcomm upstream checks.

## Decision memo

| Question | Decision |
|---|---|
| Can this Mac compile an HTP v81 context binary natively today? | **No.** No QAIRT/QNN SDK is installed, and the current Mac is `Darwin arm64`; the public SDK does not ship a macOS host build of `qnn-context-binary-generator`. |
| Can we compile v81 locally with an obtainable SDK? | **Yes, with a supported Linux or Windows host environment.** Qualcomm's public QAIRT 2.48.0 community archive contains `lib/hexagon-v81`, `libQnnHtpV81*`, and context generators for Linux/Windows, including `bin/aarch64-ubuntu-gcc9.4/qnn-context-binary-generator`. On this Mac that means first provisioning an ARM64 Ubuntu VM/container; none of Docker, Podman, Colima, Lima, or Multipass was found in `PATH`. |
| Can Qualcomm AI Hub compile/profile for v81? | **Yes.** The live inventory exposes Snapdragon 8 Elite Gen 5 / SM8850 and Samsung SM8850-AD devices with `hexagon:v81`, `framework:qnn`, FP16, and weight-sharing support. |
| Immediate path | **Use AI Hub now**, targeting `Snapdragon 8 Elite Gen 5 QRD` or `Samsung Galaxy S26 (Family)`. Add a local ARM64 Ubuntu 24.04 QAIRT 2.48 environment only if offline/reproducible local compilation is required. |

There is therefore no need to fall back for toolchain availability. If a fallback is required for scheduling or regression comparison, the nearest older target is Snapdragon 8 Elite / SM8750 / HTP v79. A v79 compile/profile is useful as a neighboring-generation proxy, but it is **not proof** of v81 compatibility, placement, latency, memory, or context-binary portability.

## 1. Qualcomm AI Hub device check

The project device-list script was run from the configured AI Hub environment inside detached tmux session `qairt-v81-gate`, as required for cloud work. Installed client: `qai-hub==0.52.0`.

Every live device matching SM8850, Snapdragon 8 Elite Gen 5, or Hexagon/HTP v81:

| AI Hub device | OS | Chipset identifiers | Hexagon | QNN |
|---|---:|---|---:|---|
| Samsung Galaxy S26 | Android 16 | `qualcomm-snapdragon-8-elite-gen5-for-galaxy`, `sm8850-ad` | v81 | yes |
| Samsung Galaxy S26 (Family) | Android 16 | `qualcomm-snapdragon-8-elite-gen5-for-galaxy`, `sm8850-ad` | v81 | yes |
| Samsung Galaxy S26 Ultra | Android 16 | `qualcomm-snapdragon-8-elite-gen5-for-galaxy`, `sm8850-ad` | v81 | yes |
| Samsung Galaxy S26+ | Android 16 | `qualcomm-snapdragon-8-elite-gen5-for-galaxy`, `sm8850-ad` | v81 | yes |
| Snapdragon 8 Elite Gen 5 QRD | Android 16 | `qualcomm-snapdragon-8-elite-gen5`, `sm8850` | v81 | yes |
| Snapdragon X2 Elite CRD | Windows 11 | `qualcomm-snapdragon-x2-elite`, `sc8480xp` | v81 | yes |

All six also advertise:

- `htp-supports-fp16:true`
- `htp-supports-weight-sharing:true`
- `framework:qnn`

**Newest exposed Snapdragon 8-series target:** Snapdragon 8 Elite Gen 5 / SM8850, represented by `Snapdragon 8 Elite Gen 5 QRD` and the Galaxy S26 family. This is the exact target generation, not an older proxy.

No compile job was submitted because this gate was explicitly read-only. The `framework:qnn` v81 targets establish that AI Hub exposes the required compilation target; the first fixed-shape submission remains the practical end-to-end confirmation.

## 2. Installed QAIRT/QNN SDK inventory

Host checks covered `$QNN_SDK_ROOT`, QAIRT/QNN/SNPE/Hexagon-related environment variables, Spotlight, and targeted filesystem searches under `/opt`, `/Applications`, `~/Library/Android`, and `~/Documents` for SDK roots, `lib/hexagon-v*`, `libQnnHtpV*`, and `qnn-platform-validator`.

Result:

- `$QNN_SDK_ROOT`: **unset**.
- Installed QAIRT/QNN SDK roots: **none found**.
- Installed `lib/hexagon-v*` target directories: **none found**.
- Installed `libQnnHtpV81*`: **none found**.

Thus there is no currently installed local SDK to inventory by version, and no current local v81 compiler/runtime payload.

Host execution constraint:

- Host is `Darwin arm64`.
- QAIRT 2.48 contains context generators for `aarch64-ubuntu-gcc9.4`, `aarch64-oe-linux-gcc9.3`, `aarch64-oe-linux-gcc11.2`, `x86_64-linux-clang`, Android, and Windows targets, but no macOS host executable.
- Qualcomm's revision history says QAIRT 2.47 added ARM-Linux as a supported development host with Ubuntu 24.04 and Python 3.12 across conversion, quantization, compilation, and accuracy-debugger tools. That makes an ARM64 Ubuntu VM the cleanest local route on Apple Silicon.

## 3. `qai-hub-models==0.57.3` target metadata

The published 0.57.3 wheel was inspected in memory; nothing was installed. Its `--device` and `--chipset` arguments are plain strings, not an `argparse` `choices=` enum. They are converted to `hub.Device(name=...)` or `hub.Device(attributes="chipset:...")`, and the live AI Hub service ultimately validates availability. The bundled `devices_and_chipsets.yaml` is therefore the package's documented target catalog, not a hard parser whitelist.

### v81-class entries in 0.57.3

| Device metadata entry | Chipset value | Alias | HTP |
|---|---|---|---:|
| Samsung Galaxy S26 / S26 Ultra / S26+ | `qualcomm-snapdragon-8-elite-gen5-for-galaxy` | `sm8850-ad` | v81 |
| Snapdragon 8 Elite Gen 5 QRD | `qualcomm-snapdragon-8-elite-gen5` | `sm8850` | v81 |
| Snapdragon X2 Elite CRD | `qualcomm-snapdragon-x2-elite` | `sc8480xp` | v81 |

So **yes, 0.57.3 explicitly contains v81-class device and chipset entries**.

### Bundled `--device` values (53)

```text
Google Pixel 3
Google Pixel 3a
Google Pixel 3 XL
Google Pixel 4
Google Pixel 4a
Google Pixel 5
Samsung Galaxy Tab S7
Samsung Galaxy S21
Samsung Galaxy S21 Ultra
Xiaomi Redmi Note 10 5G
Google Pixel 3a XL
Google Pixel 5a 5G
Samsung Galaxy A73 5G
QCS8550 (Proxy)
Samsung Galaxy S22 Ultra 5G
Samsung Galaxy S22 5G
Samsung Galaxy S22+ 5G
Samsung Galaxy Tab S8
Xiaomi 12
Xiaomi 12 Pro
Samsung Galaxy S23
Samsung Galaxy S23+
Samsung Galaxy S23 Ultra
Samsung Galaxy S24
Samsung Galaxy S24 Ultra
Samsung Galaxy S24+
Samsung Galaxy S25
Samsung Galaxy S25 Ultra
Samsung Galaxy S25+
Samsung Galaxy S26
Samsung Galaxy S26 Ultra
Samsung Galaxy S26+
Snapdragon X Elite CRD
Snapdragon X Plus 8-Core CRD
Snapdragon X2 Elite CRD
Snapdragon 8 Elite QRD
Snapdragon 8 Elite Gen 5 QRD
Snapdragon 7 Gen 4 QRD
SA8295P ADP
SA8775P ADP
SA7255P ADP
Dragonwing Q-6690 MTP
QCS8275 (Proxy)
Dragonwing RB3 Gen 2 Vision Kit
Dragonwing IQ-9075 EVK
Dragonwing IQ-8275 EVK
SA8255P ADP
SA8650P ADP
Dragonwing Q-7790
Dragonwing Q-8750
Dragonwing IQ-X5121
Dragonwing IQ-X7181
XR2 Gen 2
```

The YAML marks `Dragonwing IQ-8275 EVK`, `SA8255P ADP`, `SA8650P ADP`, `Dragonwing Q-7790`, `Dragonwing Q-8750`, `Dragonwing IQ-X5121`, `Dragonwing IQ-X7181`, and `XR2 Gen 2` as unavailable in Workbench; they are metadata/reference mappings rather than guaranteed live execution devices.

### Bundled `--chipset` canonical values, aliases, and HTP versions (36)

```text
qualcomm-snapdragon-845                 aliases: qualcomm-snapdragon-845, sdm845                         HTP v65
qualcomm-snapdragon-670                 aliases: qualcomm-snapdragon-670, sdm670                         HTP v65
qualcomm-snapdragon-855                 aliases: qualcomm-snapdragon-855, sm8150                         HTP v66
qualcomm-snapdragon-730g                aliases: qualcomm-snapdragon-730g, sm7150-ab                     HTP v65
qualcomm-snapdragon-765g                aliases: qualcomm-snapdragon-765g, sm7250                        HTP v66
qualcomm-snapdragon-865+                aliases: qualcomm-snapdragon-865+, sm8250-ab                     HTP v66
qualcomm-snapdragon-888                 aliases: qualcomm-snapdragon-888, sm8350                         HTP v68
qualcomm-snapdragon-678                 aliases: qualcomm-snapdragon-678, sm6150-ac                      HTP v66
qualcomm-snapdragon-778g                aliases: qualcomm-snapdragon-778g, sm7325                        HTP v68
qualcomm-qcs8550-proxy                  aliases: qualcomm-qcs8550-proxy, qualcomm-dragonwing-qcs8550-proxy HTP v73
qualcomm-snapdragon-8gen1               aliases: qualcomm-snapdragon-8gen1, sm8450                       HTP v69
qualcomm-snapdragon-8gen2               aliases: qualcomm-snapdragon-8gen2, sm8550                       HTP v73
qualcomm-snapdragon-8gen3               aliases: qualcomm-snapdragon-8gen3, sm8650                       HTP v75
qualcomm-snapdragon-8-elite-for-galaxy  aliases: qualcomm-snapdragon-8-elite-for-galaxy, sm8750-ac       HTP v79
qualcomm-snapdragon-8-elite-gen5-for-galaxy aliases: qualcomm-snapdragon-8-elite-gen5-for-galaxy, sm8850-ad HTP v81
qualcomm-snapdragon-x-elite             aliases: qualcomm-snapdragon-x-elite, sc8380xp                   HTP v73
qualcomm-snapdragon-x-plus-8-core       aliases: qualcomm-snapdragon-x-plus-8-core, sc8340xp             HTP v73
qualcomm-snapdragon-x2-elite            aliases: qualcomm-snapdragon-x2-elite, sc8480xp                  HTP v81
qualcomm-snapdragon-8-elite             aliases: qualcomm-snapdragon-8-elite, sm8750                     HTP v79
qualcomm-snapdragon-8-elite-gen5        aliases: qualcomm-snapdragon-8-elite-gen5, sm8850                HTP v81
qualcomm-snapdragon-7gen4               aliases: qualcomm-snapdragon-7gen4, sm7750                       HTP v73
qualcomm-sa8295p                        aliases: qualcomm-sa8295p                                        HTP v68
qualcomm-sa8775p                        aliases: qualcomm-sa8775p                                        HTP v73
qualcomm-sa7255p                        aliases: qualcomm-sa7255p                                        HTP v75
qualcomm-qcm6690                        aliases: qualcomm-qcm6690, qcm6690, qualcomm-dragonwing-q-6690   HTP v73
qualcomm-qcs8275-proxy                  aliases: qualcomm-qcs8275-proxy, qcs8275, qualcomm-dragonwing-iq-8275-proxy HTP v75
qualcomm-qcs6490                        aliases: qualcomm-qcs6490, qcs6490, qualcomm-dragonwing-qcs6490  HTP v68
qualcomm-qcs9075                        aliases: qualcomm-qcs9075, qcs9075, qualcomm-dragonwing-iq-9075  HTP v73
qualcomm-qcs8275                        aliases: qualcomm-qcs8275                                        HTP v75
qualcomm-sa8255p                        aliases: qualcomm-sa8255p                                        HTP v73
qualcomm-sa8650p                        aliases: qualcomm-sa8650p                                        HTP v73
qualcomm-qcs7790                        aliases: qualcomm-qcs7790                                        HTP v73
qualcomm-qcs8750                        aliases: qualcomm-qcs8750                                        HTP v79
qualcomm-qcs5121                        aliases: qualcomm-qcs5121                                        HTP v73
qualcomm-qcs7181                        aliases: qualcomm-qcs7181                                        HTP v73
qualcomm-qcs8450                        aliases: qualcomm-qcs8450                                        HTP v69
```

For the current work, prefer live selectors rather than relying only on YAML:

```text
--device "Snapdragon 8 Elite Gen 5 QRD"
--device "Samsung Galaxy S26 (Family)"
--chipset qualcomm-snapdragon-8-elite-gen5
--chipset sm8850
--chipset qualcomm-snapdragon-8-elite-gen5-for-galaxy
--chipset sm8850-ad
```

## 4. Phone-side v81 confirmation

Connected phone:

```text
serial: <redacted-device-serial>
product/model/device: NX809J / NX809J / NX809J
transport: USB
```

Exact output-relevant filenames from `/vendor/lib64/`:

```text
libQnnHtp.so
libQnnHtpPrepare.so
libQnnHtpV81.so
libQnnHtpV81Stub.so
```

The vendor directory did **not** list a `libQnnHtpV81Skel.so`. An additional system image copy contains:

```text
/system/etc/custom_config/app/systemUI/depth/libQnnHtp.so
/system/etc/custom_config/app/systemUI/depth/libQnnHtpV81.so
/system/etc/custom_config/app/systemUI/depth/libQnnHtpV81Skel.so
/system/etc/custom_config/app/systemUI/depth/libQnnHtpV81Stub.so
```

The `/vendor/lib64` directory permitted filename listing but denied direct file metadata/content access over the non-root ADB shell. Version strings were therefore extracted read-only from the accessible system-image copies:

| File | Embedded version evidence |
|---|---|
| `libQnnHtp.so` | `AISW_VERSION: 2.37.0`; `v2.37.0.250724175447_124859` |
| `libQnnHtpV81.so` | `lib.ver.1.0.0.libQnnHtpV81.so:1.0.0`; `AISW_VERSION: 2.37.0`; `v2.37.0.250724175447_124859`; build suffix `.b91d05aa97` |
| `libQnnHtpV81Skel.so` | `lib.ver.1.0.0.libQnnHtpV81Skel.so:1.0.0`; `AISW_VERSION: 2.37.0`; `v2.37.0.250724175447_124859`; build suffix `.b91d05aa97` |
| `libQnnHtpV81Stub.so` | `AISW_VERSION: 2.37.0`; `v2.37.0.250724175447_124859` |

This confirms that the phone is genuinely an HTP-v81 device and carries an OEM QAIRT/QNN 2.37.0 v81 runtime stack.

## 5. Qualcomm upstream release and availability check

Official Qualcomm sources report:

- Current public revision history spans QAIRT 2.34.0 through **2.48.0**.
- Latest listed release: **QAIRT 2.48.0, June 2026**.
- Qualcomm's product page links directly to the community archive `v2.48.0.260626.zip`.
- A one-byte unauthenticated range request returned HTTP `206`, `Content-Type: application/x-zip-compressed`, and total archive size `2,261,832,373` bytes. The current community archive is therefore publicly retrievable without a Qualcomm login through that direct link, although the full Software Center catalog UI asks users to log in for restricted/full catalog access.

Read-only ZIP central-directory inspection confirmed these current 2.48 payloads:

```text
lib/aarch64-android/libQnnHtpV81Stub.so
lib/aarch64-android/libQnnHtpV81CalculatorStub.so
lib/aarch64-oe-linux-gcc11.2/libQnnHtpV81Stub.so
lib/hexagon-v81/unsigned/libQnnHtpV81.so
lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so
lib/hexagon-v81/unsigned/libQairtHtpV81Skel.so
lib/hexagon-v81/unsigned/libQnnNetRunDirectV81Skel.so
bin/aarch64-ubuntu-gcc9.4/qnn-context-binary-generator
bin/x86_64-linux-clang/qnn-context-binary-generator
```

### What is the first v81-supporting QAIRT version?

There are two distinct answers, and conflating them would be misleading:

1. **Earliest v81 runtime build verified in this audit: QAIRT 2.37.0.** The connected SM8850 phone contains `libQnnHtpV81.so`, Stub, and Skel files whose embedded `AISW_VERSION` is 2.37.0.
2. **QAIRT 2.37.0 community archive is not a public local v81 SDK.** Qualcomm's still-downloadable public `2.37.0.250724` archive was inspected and contains Hexagon targets through v79, but no `hexagon-v81` directory or `libQnnHtpV81*`. The same is true of public `2.36.0.250627`.

Qualcomm's public revision-history table does not identify the release in which v81 was first added to the **community SDK**, so this audit does not invent an exact first-public version. What is proven is:

- OEM/device v81 support existed in 2.37.0.
- Public community 2.36.0 and 2.37.0 did not ship the v81 target payload.
- Public community 2.48.0 does ship the complete v81 target payload and is obtainable now.

This uncertainty does not block the port: use current QAIRT 2.48 for a local Linux toolchain, or use AI Hub's exact SM8850/v81 target immediately.

## Recommended gate outcome

**PASS for the NPU port.** The earlier concern that neither local tooling nor AI Hub could target v81 is no longer true.

1. Submit the fixed-shape graphs to AI Hub with `Snapdragon 8 Elite Gen 5 QRD` as the primary compile/profile target.
2. Repeat on `Samsung Galaxy S26 (Family)` to catch device-family packaging/runtime differences.
3. Treat the connected Nubia phone's QAIRT 2.37 OEM libraries as runtime evidence, not as a redistributable or complete host SDK.
4. If local context generation is needed, provision ARM64 Ubuntu 24.04 on the Mac and install the public QAIRT 2.48 community archive; do not attempt to run QAIRT host tools natively on macOS.
5. Keep SM8750/v79 only as a comparison/fallback row, never as an equivalence claim for SM8850/v81.

## Official/public references

- [Qualcomm AI Engine Direct SDK product page](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)
- [Qualcomm QAIRT revision history](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/general_revision_history.html?product=1601111740009302)
- [QAIRT 2.48.0 community archive](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.48.0.260626/v2.48.0.260626.zip)
- [QAIRT 2.37.0 community archive](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip)
- [QAIRT 2.36.0 community archive](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.36.0.250627/v2.36.0.250627.zip)
- [`qai-hub-models` 0.57.3 on PyPI](https://pypi.org/project/qai-hub-models/0.57.3/)
