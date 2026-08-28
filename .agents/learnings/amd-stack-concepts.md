# Stack AMD do FreeToken — Guia de Estudo

> Documento de referência: os conceitos por trás do caminho `--device tinygrad`
> (kfd, ioctls, HIP, GLSL, compilação runtime do tinygrad, RDNA3, detecção de
> hardware). Escrito para estudo — leia na ordem, as seções se constroem.

---

## 1. Os três caminhos AMD do FreeToken

| | ROCm/HIP | Vulkan (abandonado) | tinygrad/kfd (atual) |
|---|---|---|---|
| **Você escreve** | HIP C++ / Triton | GLSL à mão (37 shaders) | Python (DSL do tinygrad) |
| **Compilação** | hipcc → ROCm | glslangValidator → SPIR-V | tinygrad → GCN assembly (runtime) |
| **Execução** | ROCm runtime → kfd | Vulkan API → amdgpu | kfd ioctls → amdgpu |
| **Toolchain** | ROCm completo (pesado) | glslangValidator + libvulkan | nada (só Python) |
| **Manutenção** | kernels próprios do FreeToken | cada op = 1 shader | 1 arquivo no fork |
| **Windows** | ⚠️ (WSL2/experimental) | ✅ | ❌ (kfd é Linux-only) |

**Resumo:** o caminho atual (tinygrad) é o mais leve — zero toolchain, kernels
gerados em runtime. O preço: é Linux-only (kfd) e o caminho rápido é RDNA3-only.

---

## 2. kfd — Kernel Fusion Driver

O Linux tem **dois drivers** para GPUs AMD:

- **amdgpu** — driver gráfico (display, Vulkan, OpenGL).
- **kfd** — driver de **compute** (cálculo). É a porta de entrada para kernels
  de computação.

Fluxo do tinygrad para rodar um kernel:

```
1. open("/dev/kfd")                    → abre a porta
2. ioctl(KFD_IOC_CREATE_QUEUE)         → cria fila de execução
3. ioctl(KFD_IOC_ALLOC_MEMORY)         → aloca memória na GPU
4. submete pacotes AQL                 → Architected Queuing Language
5. ioctl(KFD_IOC_SUBMIT_QUEUE)         → dispara o trabalho
```

**AQL (Architected Queuing Language)** é a "linguagem de fila" do hardware HSA
(Heterogeneous System Architecture) — os pacotes que descrevem o kernel a rodar.

---

## 3. ioctls — a "API" do driver de kernel

**ioctl = "input/output control"** — chamada de sistema (`ioctl(2)`) que permite
um programa de usuário **falar com um driver de kernel**.

- `open("/dev/kfd")` → abre a porta.
- `ioctl(fd, KFD_IOC_CREATE_QUEUE, ...)` → "cria fila".
- `ioctl(fd, KFD_IOC_ALLOC_MEMORY, ...)` → "aloca memória".
- `ioctl(fd, KFD_IOC_SUBMIT_QUEUE, ...)` → "submete trabalho".

O tinygrad chama esses ioctls **diretamente** (via ctypes no `ops_amd.py`) —
por isso **não precisa de ROCm userspace**: fala a "língua" do kernel driver
direto, sem runtime intermediário.

---

## 4. HIP — o CUDA da AMD

**HIP = "Heterogeneous-computing Interface for Portability"** — modelo de
programação da AMD, **clone do CUDA**.

- Escreve kernels em **HIP C++** (quase idêntico a CUDA C++).
- Compila com `hipcc`.
- Roda com o **runtime ROCm** (que carrega o kernel na GPU via kfd por baixo).

**Onde aparece no projeto:**
- A branch **ROCm** (`feat/amd-rocm-gfx1100-support`) usa HIP — kernels C++
  (`dequantize_hip.cuh`, extensões compiladas com hipcc).
- O caminho **tinygrad NÃO usa HIP** — pula o runtime ROCm inteiro.

**Analogia:** HIP é escrever em C++ e compilar com um compilador; o tinygrad é
escrever em Python e o compilador gera o assembly na hora.

---

## 5. GLSL shaders — o que escrevíamos no Vulkan

**GLSL = "OpenGL Shading Language"** — linguagem C-like para programas de GPU
no OpenGL/Vulkan.

No caminho Vulkan (abandonado), escrevíamos **compute shaders** em GLSL:
- `attn_flash.comp` — flash attention.
- `gemm.comp` — multiplicação de matrizes.
- `moe_expert.comp` — experts do MoE.

Fluxo:
```
GLSL (.comp) → glslangValidator → SPIR-V (bytecode) → Vulkan API → GPU
```

Cada shader era um arquivo de texto escrito **à mão** — por isso 37 shaders e
tanta manutenção. O Vulkan compilava o SPIR-V em assembly da GPU na 1ª execução.

---

## 6. Como o tinygrad compila "shaders" em runtime

O tinygrad **não tem arquivos de shader** — **gera os kernels na hora** a partir
do Python:

```
1. Python:  x = a.matmul(b)          ← você escreve operações
2. UOps:    graph de operações       ← IR intermediário (Universal Operations)
3. TinyJit: captura o graph          ← na 1ª chamada
4. Compiler: UOps → GCN assembly     ← gera o assembly da sua GPU (gfx1100)
5. kfd:     submete via AQL packets  ← roda na GPU
6. Cache:   reutiliza nas próximas   ← mesma forma = mesmo kernel
```

**Concretamente no fork:**
- O kernel de flash attention (`kernels/amd.py`) é escrito em **Python** (DSL
  do tinygrad), não GLSL.
- Na 1ª execução, o tinygrad **compila** esse Python em assembly GCN para a
  RX 7900 XTX (gfx1100).
- Isso é o **JIT warmup** (~2.5 min no startup) — 1º request lento, seguintes
  rápidos.
- O `TinyJit` cacheia: mesma forma de entrada = mesmo kernel, sem recompilar.

**Detalhe do TinyJit (armadilha descoberta):** 1ª chamada = eager, 2ª =
capture, 3ª+ = exec. O warmup do runner roda cada JIT **2x** com shapes reais,
senão o 1º request pagava ~25s de recompile.

---

## 7. RDNA3-only — os kernels custom

O fork tem kernels custom tuneados para **RDNA3 (gfx11)**:

1. **Flash attention** — usa instruções **WMMA** (wave matrix multiply-
   accumulate) da RDNA3.
2. **GatedDeltaNet fused scan** — o scan recorrente do modelo num kernel único.
3. **GEMM com quantização custom** — WMMA + dequant inline.

Compatibilidade:

| GPU | gfx | Kernels custom |
|-----|-----|----------------|
| RX 7900 XTX | gfx1100 (RDNA3) | ✅ |
| RX 9070 (RDNA4) | gfx1200 | ❌ (layouts WMMA diferentes) |
| MI300 (CDNA) | gfx942 | ❌ (MFMA-only, wave64) |
| RX 6000 (RDNA2) | gfx1030 | ❌ (sem WMMA) |
| RX 5000 (GCN) | gfx1010 | ❌ |

---

## 8. Detecção de hardware e fallback

O gate no fork:

```python
def amd_custom_kernels_supported(device):
  # tuneado para RDNA3 (gfx11): layouts WMMA não batem com gfx12 (RDNA4)
  # ou CDNA (MFMA-only, wave64); dp4a e wave ops 32-lane não são portáveis.
  if device is None or device.split(":")[0] != "AMD": return False
  return t[0] == 11   # gfx11 = RDNA3 SOMENTE
```

Usado em **3 pontos de fallback** no modelo:

| Linha | O que | RDNA3 (gfx11) | Outro hardware |
|-------|-------|---------------|----------------|
| `model.py:190` | Atenção | flash attention custom (WMMA) | atenção padrão (correta, O(T²)) |
| `model.py:359` | GDN scan | kernel fused | loop Python 1-token-por-vez |
| `model.py:621` | generate() | chunk 256 | chunk_size=1 (lento) |

**Conclusão:** em qualquer outro hardware o modelo **RODA — correto, mas
lento**. Não crasha, não assume RDNA3.

**O código do FreeToken é agnóstico:** zero referências a `gfx1100`/`RDNA`/`7900`
no runner/engine/script. O runner usa `Transformer.from_gguf(...)` que funciona
em qualquer backend do tinygrad (AMD, CL, CPU, CUDA). O `max_context % 128` é
uma restrição do kernel decode AMD — inofensivo em outros backends.

| | Correção | Performance |
|---|---|---|
| Qualquer hardware | ✅ (fallback) | ❌ lento |
| AMD RDNA3 (7900 XTX) | ✅ | ✅ ~150 tok/s |

---

## 9. OpenCL vs kfd

| Critério | kfd | OpenCL |
|---|---|---|
| Performance (modelo híbrido) | ✅✅✅ | ❌ (fallback lento) |
| Kernels custom (flash/GDN) | ✅ | ❌ |
| Portabilidade | Linux only | ✅ (Windows/macOS/Linux) |
| Estabilidade | depende do kernel | ✅ (runtime maduro) |
| Tooling | kfd raw | ✅ (CodeXL etc.) |

**Por que kfd é quase obrigatório para ESTE modelo:** o Qwen3.5-MoE é híbrido
(GatedDeltaNet + atenção). Sem os kernels custom:
- Atenção cai para O(T²) — o hang original.
- GDN scan vira 1 token por vez — inutilizável na prática.

---

## 10. Portabilidade (Windows)

| Backend | Windows? | Nota |
|---------|----------|------|
| kfd (tinygrad AMD) | ❌ | driver de kernel Linux |
| CL (OpenCL) | ✅ | tinygrad tem backend CL |
| Vulkan | ✅ | mas tinygrad NÃO tem backend Vulkan |
| DirectML | ✅ | mas tinygrad não tem backend |
| HIP/ROCm | ⚠️ | WSL2 ou releases experimentais |

**Ironia:** o caminho Vulkan abandonado funcionaria no Windows (Vulkan é
cross-platform); o tinygrad/kfd é mais Linux-only.

---

## 11. Comparação final — por que tinygrad/kfd

| | ROCm | tinygrad kfd |
|---|---|---|
| Instalação | ROCm completo (hipcc, rocBLAS, versões casadas) | **nada** — driver amdgpu+kfd já vem no kernel Linux |
| Kernels | compilados com hipcc (binários) | gerados em runtime (Python → GCN) |
| Versão do kernel | ROCm exige versões específicas | ioctls kfd estáveis |
| Manutenção | FreeToken mantém kernels próprios | o fork mantém (1 arquivo) |

**A analogia justa:** ROCm é instalar um SDK completo para rodar um programa; o
tinygrad kfd é rodar um script que fala direto com o driver — sem SDK.

**A parte honesta:** os dois são AMD-only e o caminho rápido é RDNA3. A
vantagem do tinygrad não é portabilidade de hardware — é **zero toolchain**,
**kernels gerados em runtime**, e **menos código para manter**.
