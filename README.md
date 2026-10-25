# Auto Lip Sync

Type text, and an avatar speaks it with matching mouth shapes. Everything runs in the browser: no server, no build step. Download generated talking face videos or greenscreen mouth videos in 1080x1080p MP4 format with synchronized audio.

- Live site: https://makestreme.github.io/lip-sync
- Simpler Python (Flask) version: https://github.com/Makestreme/auto-lip-sync

## How it works

1. Text is split into sentences and converted to IPA phonemes with [phonemizer.js](https://www.npmjs.com/package/phonemizer) (espeak-ng in WASM).
2. A [Piper](https://github.com/OHF-Voice/piper1-gpl) voice (`en_US-ryan-medium`) runs in the browser via [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) and produces the audio.
3. The model is patched to also output per-phoneme durations, so each phoneme maps to an exact time.
4. Phonemes are mapped to 12 mouth images (`mouths/0.png` to `mouths/11.png`), and the current word is shown as a subtitle.

## Run locally

Serve the folder over HTTP (opening `index.html` directly will not work):

```
python -m http.server 8000
```

Then open http://localhost:8000.

## Deploy to GitHub Pages

Push this folder's contents to a repository and enable Pages from the branch root.

## Notes

- Keep sentences under about 25 words. Longer ones can turn into gibberish.
- `voices/en_US-ryan-medium.onnx` is the original Piper model with the duration tensor (`/Ceil_output_0`) added as an output, using `python -m piper.patch_voice_with_alignment`.
- `ort.env.wasm.numThreads` is 1 because GitHub Pages cannot send the headers needed for multi-threaded WASM.
- The first load downloads about 60 MB for the voice model.
- Speaking speed is the `SPEED` constant in `app.js`.

## Files

| Path | Purpose |
| --- | --- |
| `index.html` | UI and styles |
| `app.js` | Phonemization, synthesis, mouth and word timing |
| `mouths/` | Mouth shape images |
| `voices/` | Piper voice model and config |
| `favicon.svg` | Site icon |
