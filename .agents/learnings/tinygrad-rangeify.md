# tinygrad-rangeify

## Context
O gerador de grafo de kernel do tinygrad fork é o **rangeify**
(`schedule/rangeify.py:get_kernel_graph` → `run_rangeify` em
`schedule/indexing.py`). Extensões simbólicas (variable-T, start_pos bound)
expondo 2 invariantes mal sucedidas: o `ended` comprehension (o IndexError do
broadcast OOB) e o raise "REDUCE has no ranges" no
`convert_reduce_to_reduce_with_ranges`.

## Hardest decision
A causa raiz do IndexError: o `broadcast_axes(x.shape, c.shape)` devolve
índices no **rank completo** do consumidor, mas o `range_map[c][0]` é
PÓS-drop — o `broadcast_rngs` derruba os `nleft` primeiros ranges para EXPAND
(`rngs[len(arg):]`) e o bitcast/reshape merge derruba os TRAILING — o índice
full-rank sai do tuple. Fix: guardar os índices OOB.

## Alternatives rejected
- O fix no grafo-modelo (pad/pin dos shapes): o 16K não é controlável pelo
  modelo; o MTP-mock é mock.
- O try/except em volta do run_rangeify com o fallback legado: o scheduler
  legado não existe ativo no fork (o rangeify É o gerador).
- O return None sozinho no REDUCE sem ranges: quebra o downstream bitcast
  ("unsupported size in bitcast") — como o REDUCE não convertido sai do
  formato que o bufferize espera; o Plan B (sintetizar os ranges do shape do
  src com `ctx.new_range`) funciona.

## Least confident
- O guard do índice pode ter falhas latentes nos cases de drop LEADING
  (não reproduzidas no crash — o crash real é o drop TRAILING do bitcast
  merge). O sweep gating (o delta do snapshot) cobre parcialmente.

## Reuse
`tinygrad/schedule/indexing.py` (o fix), `tinygrad/llm/kernels/amd.py`
(`kv_q8_quantize_batched`, `expert_linear`), `test/unit/test_rangeify_repro.py`
(os pins). Read antes de tocar o scheduler do tinygrad fork ou os shapes
simbólicos dos kernels.
