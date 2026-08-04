# Multilingual/VAD mobile UI visual QA

Date: 2026-07-23 22:50 CST
Status: host-browser pass; physical Android WebView validation remains

The merged demo UI was served directly from the packaged asset directory and
inspected in the in-app browser at 390 x 844 and 320 x 720 CSS pixels.

Observed:

- no horizontal overflow at either phone width;
- the three response modes, Push-to-talk/Hands-free switch, language selector,
  history budget/reset control, microphone action, and footer remain reachable;
- selecting Mandarin changes the visible route to
  `Mandarin Chinese · 100% · explicit user selection`;
- selecting Hands-free changes the visible state to
  `Hands-free stopped · Java CPU VAD · barge-in off` and the primary action to
  `Start hands-free`;
- the 320-pixel layout wraps controls vertically without clipping them outside
  the document width;
- the browser recorded no JavaScript warning or error entries.

This check validates responsive layout and basic control state only. It does
not simulate the Android native bridge, microphone permission, acoustic VAD,
TTS voices, model service, or physical playback timing.
