# Copilot stream admission

Symptom: VS Code waits for several minutes, then logs `Response contained no choices`,
while GPU and PLE counters continue advancing.

Cause: an OpenAI `StreamingResponse` has already committed HTTP 200 headers. An error-only
SSE frame has no `choices`, which Copilot rejects. Serial Qwen engines also have a scheduler
pending queue; counting queued requests as running returns 429 while the engine can still
process them.

Fix: emit data-bearing heartbeat chunks during long prefill, send terminal error chunks with
`choices` plus `[DONE]`. Serial engines admit a small bounded pending queue and return HTTP
429/`Retry-After` only after queue capacity is exhausted. Keep request deadlines finite and
preserve configured prefill chunk size; do not hide slow prefill by removing watchdog.
