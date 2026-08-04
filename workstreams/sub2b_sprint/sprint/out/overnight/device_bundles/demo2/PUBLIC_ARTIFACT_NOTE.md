# Public artifact note

The accepted Demo 2 runtime APK is intentionally omitted from the public Git
snapshot because its 121,215,600-byte size exceeds GitHub's 100 MiB per-file
limit.

Its accepted identity remains recorded in `accepted_apk_pair.json`:

- Package: `ai.liquid.lfmdemo.runtime`
- File: `app-runtime-accepted-33bf0d6d.apk`
- SHA-256: `33bf0d6d284d6eaad96308f22a8d10c3879ba6d0b792f893486be03f0a17e244`

The public checkout includes the accepted main APK, manifests, reports, and
device evidence. Restoring the exact accepted two-APK configuration still
requires a locally archived or rebuilt runtime APK with the hash above.
