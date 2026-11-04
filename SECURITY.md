# Security and runtime boundaries

Static site, no accounts, cookies, analytics, payment paths, keys, writable server or text storage. User text is used only within a dedicated browser worker and never sent to a server. Output uses textContent, not HTML injection. The only third-party request is a pinned ONNX Runtime WASM binary from jsDelivr; a hard-coded SHA256 must match before execution. Runtime JS and model are same-origin assets. See docs/runtime-integrity.json and bundled MIT license.

Content Security Policy restricts scripts, workers, styles and base URLs to this origin, with wasm execution allowed. connect-src allows the site and jsDelivr. No forms, objects or inline script. Download/network failure yields no result, not a fallback score. Worker timeout is 90 seconds; Clear terminates it. Inputs are bounded to 10,000 characters and the model receives 128 tokens. Scores do not prove authorship and are prohibited as a basis for punitive decisions.

Server authentication, authorization, database RLS, queues, payment idempotency and server rate limits are not applicable: this project has no writable backend. Static hosting/CDN caching is provider-controlled. This is a noncommercial portfolio experiment; do not submit sensitive text on a shared or compromised browser. Device/WASM compatibility is not universal.

Reproduction code downloads fixed revision datasets and a pretrained model from Hugging Face. Run it only in an environment where those dependencies are acceptable. The reference corpus is not shipped as raw prose. Report issues through the repository, without private text or credentials.

## Scanner triage

The manifest JSON files contain literal SHA256 text hashes, not HockeyApp or other credentials. Generic token detectors can match substrings of these public digests. Dynamic-url download warning is constrained to a hard-coded Hugging Face HTTPS origin, revision and two literal filenames, not user input. DataLoader pin_memory=False is intentional CPU-only behavior, not a security concern. Runtime/model vendor binaries are verified separately by SHA256 and licenses rather than language-code scans. No claim of universal vulnerability absence.
