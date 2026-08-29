# tinygrad-moe-capture-nan

## Context
O fused expert-decode GEMM (f73a3eda0) corrompe grafos de decode TinyJit em
runtime sp >= ~16K: logits NaN -> multinomial mata o scheduler. O greedy
(argmax) engole NaN silenciosamente - TODOS os benches eram content-blind.
O path genérico weight[sel] no mesmo grafo é 100% finito (16K/61K).

## Hardest decision
O diagnóstico por stats lazy (nan/inf/absmax por camada/sub-parte, inclusas
no grafo capturado e realizadas em UM realize fora do JIT) classificou o
primeiro NaN em UMA passada de GPU - a alternativa (variantes sequenciais)
sofre state-poison (o exec NaN escreve estado NaN).

## Alternatives rejected
- Static full-coverage do flash decode (remover o slice [:live] variável +
  cap 8192): MMU fault em 61K (o kernel não estava pronto); revertido.
- Re-captura em sp grande: refutado (capture@16256 quebrou igual em exec@16385).
- moe-tensorq8: quebra igual - o Q8-quantize não era o culpado.

## Least confident
A mecânica exata no planner de kernel-graph do fork (o porquê do fused vs
genérico divergirem no captured-exec) - aberta, evidência em
.plans/decode-nan-captured-jit/.

## Reuse
TinyJit + kernels custom: SEMPRE validar o conteúdo (não só tok/s) do
captured-exec; os benches greedy não pegam NaN. scripts/probe-decode-nan.py
é o harness. O flag MOE_FUSED_DECODE ampère o restore.
