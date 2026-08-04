# Isolated QAIRT 2.45 Supertonic lane

- Date: 2026-07-30
- Target: Snapdragon 8 Elite Gen 5 QRD / SM8850 / SoC model 87 / HTP v81
- Candidate package: `ai.liquid.lfmdemo.ttsprobe.qairt245`

## Outcome

The isolated QAIRT 2.45 lane is **ready for remote artifact acquisition** and
is the fastest realistic path to a first four-component Supertonic NPU canary.
All four existing Qualcomm AI Hub compile jobs are successful, their target
models are still authenticated, and an official Android QAIRT runtime archive
matches their exact build: `2.45.0.260326154327`.

This is not an APK or phone result. No context binary was downloaded to this
Mac, no ADB or phone mutation occurred, and no accepted package or QAIRT 2.42
file was changed. The lane must remain a separate package, UID, process, native
library namespace, and app-private storage tree.

Machine-readable inputs and controller:

- [qairt245_supertonic_isolated_spec.json](qairt245_supertonic_isolated_spec.json)
- `qairt245_supertonic_hub_state.json` (retained in the non-public raw-evidence archive)
- [qairt245_supertonic_hub_acquire.ipynb](qairt245_supertonic_hub_acquire.ipynb)
- [`scripts/qairt245_supertonic_hub_acquire.py`](../../../scripts/qairt245_supertonic_hub_acquire.py)
- [`scripts/test_qairt245_supertonic_hub_acquire.py`](../../../scripts/test_qairt245_supertonic_hub_acquire.py)

Current acquisition trust roots:

| Control | SHA-256 |
|---|---|
| `qairt245_supertonic_isolated_spec.json` | `79cf01d8bbc9ba783d41196edce0f830a49c1d3e50549a222fdf09aacb3cf8ce` |
| `qairt245_supertonic_hub_state.json` | `8e6be47a6da10efd2dc1ae667dba3c54119c070cff6fb30ae80c091a6eca71c6` |
| `scripts/qairt245_supertonic_hub_acquire.py` | `2b7b072989c05de40911d20b1c0ca9764c3a9a16a1eada7bfb06d1d5df2292df` |
| `qairt245_supertonic_hub_acquire.ipynb` | `ef7b1ee77ff7cf8a3115a8a24640c10c98ed43c249acea97ec659d4f79f5ed20` |
| `scripts/test_qairt245_supertonic_hub_acquire.py` | `36d290ed02a740481108e25afe277076336b1ab04f2b8be7f9d3b22e734c21f5` |

The historical Hub state remains byte-for-byte unchanged and therefore records
the controller and spec hashes that performed the authenticated metadata
observation. The current controller and spec changed only for local
acquisition/resume and artifact-lock verifier hardening; none of the Hub job,
model, tensor, target, object, runtime, or member identities changed. The
notebook pins the historical observation and current controls independently
instead of rewriting the authenticated observation after the fact.
The `metadata` command now requires an explicit `--state` destination so an
operator cannot accidentally overwrite that pinned historical file, and it
rejects the historical filename even when supplied explicitly.

## Authenticated Hub artifacts

The pinned `qai-hub==0.52.0` client re-read every job, target model, tensor
contract, exact target device, framework build, and scoped download object.
The controller rejects any credential profile whose API or web endpoint is not
exactly `https://workbench.aihub.qualcomm.com`; a lookalike or staging service
cannot satisfy the metadata trust boundary.
The server exposes no SHA-256 for these objects, so SHA-256 remains a mandatory
Linux/Colab acquisition gate.

| Component | Compile job | Target model | Format | Bytes | Opaque object ETag |
|---|---|---|---|---:|---|
| Duration predictor | `jp2v287x5` | `mq3l4wjrq` | QNN context binary | 1,998,848 | `0fcbf3f0cd68add465a3e9e8bf472e19` |
| Text encoder | `j5mdj79d5` | `mq3l410lq` | QNN context binary | 18,841,600 | `3d698f91574cc3d64abe8b2f2bf3d427` |
| Vector estimator | `j5w1o2k6g` | `mqvwv53jn` | QNN context binary | 132,395,008 | `1edb15efc14de908a66d85b1206ddf30` |
| Vocoder | `jgo4ykxx5` | `mngx4ve6q` | QNN context binary | 35,282,944 | `d4b98f9801348409432532d0cc623433` |

Total target-model payload: **188,518,400 bytes**. The spec also pins each
object filename, last-modified time, and object version ID. ETags are treated
as opaque locators, not as MD5 or cryptographic trust roots.

Every target reports:

- QAIRT `2.45.0.260326154327`;
- backend `HTP`;
- Hexagon `v81`;
- SoC model `87`;
- optimization level `3`;
- target type `QNN_CONTEXT_BINARY`.

The earlier exact-2.42 Hub submission remains rejected before job creation;
the account currently advertises 2.45, 2.47, and 2.48, not 2.42.

## Exact matching Android runtime

Official archive:

`https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.45.0.260326/v2.45.0.260326.zip`

- HTTP bytes: `1,657,471,677`;
- opaque ETag: `c54aff28c456271cc0fa4b97b9d04b17`;
- object version ID: `9iP6qAPad6HDRH7Pl7izwqeq_nw.jxY.`;
- last modified: `Fri, 03 Apr 2026 03:30:18 GMT`;
- full archive SHA-256: not yet known; computed by `acquire` on Linux/Colab.

A bounded ZIP-index inspection fetched about 1.8 MB, not the SDK. The internal
`sdk.yaml` is SHA-256
`3f1e62262514b454ec66ad9095a1ef12f9642341f0007952a36f79997ce8edf3`
and states:

```text
product: QAIRT
version: 2.45.0
build_id: 260326154327
qnn_backend_api_version: 2.18.0
android-ndk: r26c
```

This joins to the Hub context build exactly as
`2.45.0.260326154327`. A second bounded pass downloaded and decompressed only
the required members, about 48 MB compressed total, and pinned these hashes:

| Role | Member | Bytes | SHA-256 |
|---|---|---:|---|
| Canary frontend | `bin/aarch64-android/qnn-net-run` | 5,051,616 | `812a0f98f8bcd2ecc39c8a2462e39ad0f031dfaa565b3d095e4475b5484ecba3` |
| HTP backend | `lib/aarch64-android/libQnnHtp.so` | 2,952,008 | `b5b36b2d5e0cc352b5e5f468f6f668b1585d515854fc1c5890b1365047a36be1` |
| Net-run extension | `lib/aarch64-android/libQnnHtpNetRunExtensions.so` | 957,864 | `0d29d3dac7d82d8eb5bdc47a0649a7d6a5f5b4a2fbc52b2f975b6ae2c16b6213` |
| HTP prepare | `lib/aarch64-android/libQnnHtpPrepare.so` | 87,739,504 | `d0001d629c6d7ded14f2c850e93983b87589cd5748724bb9aaf56f7adb2b338a` |
| v81 Stub | `lib/aarch64-android/libQnnHtpV81Stub.so` | 803,416 | `e97898d6772288e202e9cf988b80b39b8336f3dca8b99ee3075d234093582e3c` |
| QNN System | `lib/aarch64-android/libQnnSystem.so` | 3,635,400 | `f08e9ddc18a40719839cce8bbfb3afd3aef06f5ad486b21934a1c4568c9a6a48` |
| v81 unsigned runtime | `lib/hexagon-v81/unsigned/libQnnHtpV81.so` | 12,108,736 | `b4610877765705aa12bdb27fb8d7e8f87692bfa2bbf86a0a9e525d3159cf1c1d` |
| v81 unsigned Skel | `lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so` | 11,788,496 | `57330ed3f7846b94181b25b7a87bc3fe5dc71033766f2d5630520a1400477c97` |
| License | `LICENSE.pdf` | 147,577 | `ec1dccfdcba5c6e64126e84199b8362bf4999107bfa567ebe831dbb4c461692b` |
| Notice | `NOTICE.txt` | 129,238 | `c8a22fa8b9ed3e9266c4a3366965474cef5cb147ee2a1f33e488bbb24a071c09` |

The runtime member paths, compressed sizes, and ZIP CRC32 values are also
pinned in the spec and will be rechecked after the full Colab download.

## Resume-safe Colab acquisition

The controller has four commands:

```bash
# Offline; safe on this Mac.
python3 scripts/qairt245_supertonic_hub_acquire.py plan

# Authenticated metadata plus bounded runtime index/member reads; no full
# context or runtime-archive download.
python scripts/qairt245_supertonic_hub_acquire.py metadata \
  --state /content/qairt245-fresh-metadata.json

# Linux/x86_64 only. Use a persistent Google Drive workspace.
python scripts/qairt245_supertonic_hub_acquire.py acquire \
  --workspace /content/drive/MyDrive/lmf-demo/qairt245-supertonic

# Rehash every locked context, runtime member, and the full archive. The lock
# hash must come from the trusted acquire output or a separately pinned record,
# never by recomputing it from the workspace being checked.
python scripts/qairt245_supertonic_hub_acquire.py verify \
  --workspace /content/drive/MyDrive/lmf-demo/qairt245-supertonic \
  --expected-lock-sha256 <trusted-64-hex-lock-sha256>
```

The Colab session needs `qai-hub==0.52.0` and a session-provisioned standard
AI Hub credential/profile. Credentials must not be copied into this repository
or the persistent artifact lock.

The hash-pinned notebook above is the preferred operator path. It mounts a
dedicated Google Drive root for durable controls, checkpoints, downloads, and
the final lock, while keeping the AI Hub profile, fresh metadata output, and
redacted controller logs only under ephemeral `/content`. It prompts through
`getpass`, creates the profile with mode `0600` before writing any token bytes,
passes its path only through
`QAIHUB_CLIENT_INI`, captures rather than echoes controller output, and removes
all session material in a `finally` block. It deliberately does not use
`qai-hub configure`: client `0.52.0` prints the saved profile, including its
token. No credential is placed in Google Drive, a command argument, notebook
output, repository file, acquisition state, or artifact lock. Raw exceptions
from external Hub/download libraries are deliberately omitted from persistent
failure state so a signed URL or scoped credential cannot be captured there;
diagnostic detail remains only in the ephemeral, redacted session log.

The notebook installs only the pinned `qai-hub==0.52.0` client and its declared
dependencies. Both the notebook and controller reject any host other than
Linux/x86_64 before acquisition. No network command from the notebook was run
on this Mac; the committed notebook was validated structurally and by compiling
every code cell.

`acquire` refuses macOS and non-x86_64 hosts. It performs a fresh metadata and
download-authorization check and freshly downloads all four contexts in every
acquisition run. Completed context checkpoints are deliberately not reused:
their SHA-256 values do not exist in the repository trust root yet, so a
checkpoint cannot authenticate its own unknown hash. A prior identical context
is retained only when it byte-matches the new authenticated download; a
different prior file is quarantined. The controller resumes the large public
runtime download over HTTPS with the pinned object ETag, then validates the
exact pinned central-directory hash plus every allowlisted runtime member's
path, size, compressed size, CRC32, and SHA-256 before accepting either a fresh
or completed archive checkpoint. It computes all resulting SHA-256 values,
extracts only the pinned Android members, and writes
`qairt245_supertonic_artifact_lock.json` atomically. An APK build is forbidden
until that lock exists, its SHA-256 is recorded outside the workspace trust
boundary, and `verify` passes with that exact trusted hash.

The artifact-lock verifier is fail-closed rather than treating workspace JSON
as its own authority. Standalone `verify` requires the caller-supplied SHA-256
of the complete lock from the trusted acquisition output or a separately pinned
external record, and rejects a missing or different value before consuming the
lock. It then requires the exact top-level schema; the exact four-component set
and order; each context path, byte count, target-model ID, authenticated object
filename/ETag/version/time, and acquisition-state record; the pinned ZIP and
runtime-member content contract; the protected package hashes and QAIRT build;
the exact target, candidate package, controller, spec, metadata fingerprint,
and completed Linux/x86_64 acquisition-state identity; and no extra or
duplicate records. This closes both the earlier tiny-file counterfeit and the
residual exact-sized/self-hashed checkpoint substitution path.

Twenty-six focused offline tests pass. Regressions cover arbitrary self-hashed
tiny contexts/runtime/members, an exact-sized forged context in a forged
complete checkpoint/lock, a self-declared non-ZIP runtime archive checkpoint,
missing and wrong trusted lock hashes, the clean acquire-to-verify lock-hash
handoff, non-Qualcomm Hub endpoints, HTTPS downgrade redirects, external-error
credential redaction, missing and extra contexts, missing/extra/duplicate
runtime members, swapped context records, fake protected hashes, unexpected
lock keys, protected historical-metadata output, acquisition-state controller
drift, and post-lock file tampering.
Both Python controls pass `ruff check`, `ruff format --check`, and
`python -m py_compile`.

## Isolated package contract

The first candidate must use all of these boundaries:

- package: `ai.liquid.lfmdemo.ttsprobe.qairt245`;
- dedicated process: `ai.liquid.lfmdemo.ttsprobe.qairt245:qairt245`;
- a new UID and app-private model/runtime storage;
- only the exact QAIRT 2.45 libraries listed above;
- no accepted ORT-QNN/QAIRT 2.42 AAR, library, context, directory, or process;
- a whole-APK SHA-256 recorded outside the candidate UID before installation;
- no neural CPU fallback;
- candidate-UID HTP/CDSP, FastRPC, DMA, output, latency, and cleanup evidence
  before any integration claim.

`qnn-net-run` is pinned as a private engineering canary frontend. A minimal JNI
bridge compiled against the exact QAIRT 2.45 headers is the cleaner application
frontend. Android executable packaging and JNI source are deliberately held
until the four context hashes exist, preventing an unverified object from
entering an APK.

Qualcomm's license permits relevant object-code distribution only as
incorporated into the developer application and not as a standalone runtime
redistribution. Keep the SDK and extracted libraries outside the public repo;
perform a distribution review before sharing any candidate APK.

## 2.45 versus exact 2.42

| Lane | First-canary speed | Safety boundary | Main risk |
|---|---|---|---|
| Isolated 2.45 | **Fastest.** Four compiled contexts and exact runtime already exist. | Safe only in a new package/UID/native namespace. | New APK/JNI integration and phone compatibility still unproven. |
| Exact 2.42 remote AOT | Slower and more conversion work; Hub rejects 2.42. | Best eventual fit for the already accepted app-owned 2.42 runtime if successful. | Converter/operator/toolchain failure, especially vector estimator and vocoder. |

Therefore 2.45 should be the first canary, while exact-2.42 remote AOT remains
the preferred eventual convergence route if its independent conversion lane
succeeds. Neither lane may borrow the other's contexts or native libraries.

## Remaining gates

1. Run `acquire` in Linux/Colab and persist the returned artifact-lock SHA-256,
   four context SHA-256 values, and full archive SHA-256 outside the workspace
   being verified.
2. Run `verify --expected-lock-sha256 <trusted value>`; reject any lock, size,
   ETag/version identity, ZIP/member, or file-hash drift.
3. Build a minimal isolated candidate APK only from the verified artifact lock.
4. Run component-by-component phone validation before the accumulated chain.
5. Prove NPU placement and multilingual output quality under the candidate UID.
6. Recheck both accepted APK hashes and accepted runtime health after stopping
   the candidate.

Until those gates pass, the lane status is
`metadata_authenticated_artifacts_not_downloaded`.
