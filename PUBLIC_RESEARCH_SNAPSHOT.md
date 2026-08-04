# Public research snapshot

This branch is a clean-root publication snapshot of the on-device multilingual
speech research and Nubia phone-demo work. It intentionally does not inherit
the private sprint history or local assistant/session state.

Included:

- Android implementation, tests, experiment specifications, and reusable tools
- Curated research results, bounded claims, and phone-measured evidence
- The normalized Demo 2 film and the accepted main application APK
- The latest selected-route latency/preload handoff for English, Mandarin, and
  Russian, including the disclosed short-Russian playback limitation

Excluded:

- Coding-assistant configuration, session state, prompt packets, agent
  playbooks, lane briefs, task graphs, and generated assistant research drafts
- Personal workstation paths and the phone's unique ADB serial; publication
  copies use neutral placeholders while retaining measurement values
- `app-runtime-accepted-33bf0d6d.apk`, because its 121,215,600-byte size
  exceeds GitHub's 100 MiB per-file limit
- `DEMO1_IT_FITS.mp4`, rollback APKs, mutable progress logs, and bulky raw
  captures; their curated receipts and hashes remain where applicable

The omitted runtime APK's package identity, byte count, and SHA-256 remain in
`workstreams/sub2b_sprint/sprint/out/overnight/device_bundles/demo2/accepted_apk_pair.json`
and the adjacent `PUBLIC_ARTIFACT_NOTE.md`. Exact restoration requires a
separately archived artifact matching that hash.

This snapshot replaces the earlier public root so assistant-only artifacts and
accidentally published local identifiers are not retained in public history.
