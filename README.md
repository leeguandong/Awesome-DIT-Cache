<h1 align="center">
Awesome-Dit-Cache
</h1>
<p align="center">
<b>An Awesome Collection for Feature Caching in Diffusion / DiT Models / 收集和梳理扩散模型与 DiT 的 Feature Cache 加速方法</b>
</p>
<p align="center">
  <a href="https://github.com/leeguandong/Awesome-Dit-Cache/stargazers"> <img src="https://img.shields.io/github/stars/leeguandong/Awesome-Dit-Cache.svg?style=popout-square" alt="GitHub stars"></a>
  <a href="https://github.com/leeguandong/Awesome-Dit-Cache/issues"> <img src="https://img.shields.io/github/issues/leeguandong/Awesome-Dit-Cache.svg?style=popout-square" alt="GitHub issues"></a>
  <a href="https://github.com/leeguandong/Awesome-Dit-Cache/forks"> <img src="https://img.shields.io/github/forks/leeguandong/Awesome-Dit-Cache.svg?style=popout-square" alt="GitHub forks"></a>
</p>

> A curated list of feature caching / training-free acceleration methods for Diffusion Models and Diffusion Transformers (DiT), covering image, video, and flow-matching generators.

本项目旨在收集和梳理 **扩散模型（UNet / DiT / Flow Matching）推理加速** 中的 **Cache 类方法**——包括 Timestep-Adaptive、Layer-Adaptive、Predictive (Cache-then-Forecast)、Token-Level、Frequency-Aware、CFG-Level、Video-DiT 等主流范式。聚焦训练无关（training-free）的特征缓存与近邻近似方法。

如果本项目能给您带来一点点帮助，麻烦点个⭐️吧～

同时也欢迎大家贡献本项目未收录的论文、开源实现。提供新的仓库信息请发起 PR，并按照本项目的格式提供仓库链接、arXiv 编号、会议、简介等信息，感谢～

## About

**Why this repo / 为什么做这个仓库**

过去两年里，Diffusion / DiT 推理加速领域的 Cache 类方法井喷式出现——从 2023 年 DeepCache 把"特征复用"第一次系统化，到 2025 年 TaylorSeer、HiCache、FoCa 把它升级成"数值积分式预测"，再到 2026 年 SeaCache / SpectralCache / LayerCache 把频域、层深度、JVP 前向一并拉进调度框架，方法演化已经跨越了 **"复用 → 调度 → 预测 → 多轴混合"** 四个阶段。但这个方向缺少一个**统一的中文索引**：论文分散在 CVPR / ICLR / ICCV / NeurIPS / arXiv，Video DiT 与 Image DiT 的工作被割裂收录，新老 baseline 对比困难。

这个仓库就是为了填这个缺口：
- **以"调度粒度"为主轴**（Static → Timestep → Layer → Predictive → Token → Frequency → CFG → Hybrid），把 2023–2026 的代表性方法一次性摆到同一张表里。
- **双语**介绍，论文 + arXiv + 代码链接齐全，方便检索与追溯。
- **覆盖范围** = Image DiT（FLUX / SD3 / PixArt / Qwen-Image）+ Video DiT（CogVideoX / HunyuanVideo / Wan / Open-Sora）+ Flow Matching，不只是狭义的 image 场景。
- 顺带记录两个本人主导的工作作为对应范式的代表：**SpectralCache**（频域 × 时步 × 误差预算的 Hybrid 代表）与 **LayerCache**（CVPR 2026，层异质速度 + JVP 预测）。

**Scope / 收录边界**

✅ 收录：training-free 特征缓存、激活复用、时步/层/token/频域/CFG 维度的复用策略、预测式缓存（Taylor/Hermite/ODE 数值法）、视频 DiT cache。
❌ 暂不收录：量化（FP8/INT8）、蒸馏（step distillation / consistency models）、注意力内核优化（SageAttn、FlashAttn）、并行推理（xDiT / USP） —— 这些虽然常与 cache 叠加使用，但各自有独立的加速机理，另起仓库更合适。

**Maintainer**

由 [@leeguandong](https://github.com/leeguandong) 维护，欢迎 issue / PR。如果你在做 cache 相关工作希望被收录，请提供：论文 arXiv、代码仓库、目标模型、加速比、**两轴归属**（缓存粒度 §2 + 调度策略 §3）。

## 目录

- [About](#about)
- [1. 方法汇总](#1-方法汇总)
  - [1.1 方法全景表](#11-方法全景表)
  - [1.2 演化时间线](#12-演化时间线)
- [2. 按缓存粒度分类（What is cached）](#2-按缓存粒度分类what-is-cached)
  - [2.1 Step Cache（整步输出）](#21-step-cache整步输出)
  - [2.2 Block Cache（Transformer Block 输出）](#22-block-cachetransformer-block-输出)
  - [2.3 Attention Cache（注意力模块）](#23-attention-cache注意力模块)
  - [2.4 MLP / FFN Cache](#24-mlp--ffn-cache)
  - [2.5 Token Cache（token 子集）](#25-token-cachetoken-子集)
  - [2.6 Frequency-Band Cache（频带分解）](#26-frequency-band-cache频带分解)
  - [2.7 CFG-Branch Cache](#27-cfg-branch-cache)
  - [2.8 Residual Cache（层间残差）](#28-residual-cache层间残差)
  - [2.9 缓存粒度 × 调度策略 交叉矩阵](#29-缓存粒度--调度策略-交叉矩阵)
- [3. 按调度策略详述（How to decide）](#3-按调度策略详述how-to-decide)
  - [3.1 Static Caching（固定调度）](#31-static-caching固定调度)
  - [3.2 Timestep-Adaptive（时步自适应）](#32-timestep-adaptive时步自适应)
  - [3.3 Layer-Adaptive（深度自适应）](#33-layer-adaptive深度自适应)
  - [3.4 Predictive / Cache-then-Forecast（预测类）](#34-predictive--cache-then-forecast预测类)
  - [3.5 Token-Level / Granularity（细粒度）](#35-token-level--granularity细粒度)
  - [3.6 Frequency-Aware（频域类）](#36-frequency-aware频域类)
  - [3.7 CFG-Level Caching](#37-cfg-level-caching)
  - [3.8 Video DiT Cache（视频专用）](#38-video-dit-cache视频专用)
  - [3.9 Hybrid / Multi-Dimensional（混合类）](#39-hybrid--multi-dimensional混合类)
- [4. 测评](#4-测评)
  - [4.1 常用评测指标](#41-常用评测指标)
  - [4.2 基线模型](#42-基线模型)
  - [4.3 常用 Benchmark](#43-常用-benchmark)
- [5. 工程与工具](#5-工程与工具)
- [6. 相关综述](#6-相关综述)
- [Star History](#star-history)
- [License](#license)

## 1. 方法汇总

### 1.1 方法全景表

| 方法 | 会议/年 | 目标模型 | 范式 | 典型加速 | arXiv | 代码 |
|------|--------|---------|------|---------|-------|------|
| **DeepCache** | CVPR 2024 | UNet (SD 1.5/2.x) | Static + 时步 | ~2.3× | [2312.00858](https://arxiv.org/abs/2312.00858) | [horseee/DeepCache](https://github.com/horseee/DeepCache) |
| **FasterDiffusion** | NeurIPS 2024 | UNet | Static (encoder skip) | ~1.8× | - | - |
| **T-GATE V1/V2** | 2024 | SD / PixArt / LCM | 阶段式 (cross-attn freeze) | ~1.5× | [2404.02747](https://arxiv.org/abs/2404.02747) | [HaozheLiu-ST/T-GATE](https://github.com/HaozheLiu-ST/T-GATE) |
| **FORA** | 2024 | DiT | Static 固定区间 | ~1.8× | [2407.01425](https://arxiv.org/abs/2407.01425) | - |
| **Δ-DiT** | 2024 | DiT | Static (residual cache) | ~1.6× | [2406.01125](https://arxiv.org/abs/2406.01125) | - |
| **Block Cache / Cache Me if You Can** | CVPR 2024 | UNet | Layer-Adaptive 阈值 | ~1.8× | [2312.03209](https://arxiv.org/abs/2312.03209) | - |
| **PAB** (Pyramid Attention Broadcast) | ICLR 2025 | Video DiT (Open-Sora/Latte) | Static × attention 类型 | ~10.6× FPS | [2408.12588](https://arxiv.org/abs/2408.12588) | [NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys) |
| **FasterCache** | ICLR 2025 | Video DiT | Hybrid (feature + CFG) | ~1.67× | [2410.19355](https://arxiv.org/abs/2410.19355) | [Vchitect/FasterCache](https://github.com/Vchitect/FasterCache) |
| **AdaCache** | 2024 | Video DiT | Layer-Adaptive (per-video) | ~4.49× | [2411.02397](https://arxiv.org/abs/2411.02397) | [adacache-dit](https://adacache-dit.github.io/) |
| **TeaCache** | CVPR 2025 | DiT / Video DiT | Timestep-Adaptive (阈值) | ~2.1× | [2411.19108](https://arxiv.org/abs/2411.19108) | [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) |
| **FBCache** (First-Block Cache) | 2024 | DiT | Timestep-Adaptive (首层触发) | ~1.87× | - | [chengzeyi/ParaAttention](https://github.com/chengzeyi/ParaAttention) |
| **HarmoniCa** | 2024 | UNet | Layer-Adaptive (learning) | ~1.7× | [2410.01723](https://arxiv.org/abs/2410.01723) | - |
| **MagCache** | 2025 | DiT | Timestep (幅值定律) | ~2.0× | [2506.09045](https://arxiv.org/abs/2506.09045) | [Zehong-Ma/MagCache](https://github.com/Zehong-Ma/MagCache) |
| **EasyCache** | ICCV 2025 | Video DiT | Timestep (runtime self-correct) | 2.1–3.3× | [2507.02860](https://arxiv.org/abs/2507.02860) | - |
| **LazyDiT** | AAAI 2025 | DiT | Timestep (learned skip) | ~1.9× | [2412.12444](https://arxiv.org/abs/2412.12444) | - |
| **Chipmunk** | 2025 | DiT | Timestep (稀疏增量) | ~2.5× | [2506.03275](https://arxiv.org/abs/2506.03275) | - |
| **ToCa** | ICLR 2025 | DiT | Token-Level | ~1.5× | [2410.05317](https://arxiv.org/abs/2410.05317) | [Shenyi-Z/ToCa](https://github.com/Shenyi-Z/ToCa) |
| **DuCa** (Dual Feature Cache) | 2024 | DiT | Token × Layer 双层 | ~1.9× | [2412.18911](https://arxiv.org/abs/2412.18911) | - |
| **FastCache** | 2025 | DiT | Token 线性近似 | ~4.5× | [2505.20353](https://arxiv.org/abs/2505.20353) | - |
| **DiCache** | 2025 | DiT | shallow probe 自触发 | ~2.3× | [2508.17356](https://arxiv.org/abs/2508.17356) | - |
| **DBCache** | 2025 | DiT | Probe-Decide-Correct | ~2.0× | - | - |
| **Skip-DiT** | ICCV 2025 | DiT | Long-Skip-Connection + cache | 1.5–2× | [2411.17616](https://arxiv.org/abs/2411.17616) | - |
| **TaylorSeer** | ICCV 2025 | DiT | Predictive (Taylor) | ~2.4× | [2503.06923](https://arxiv.org/abs/2503.06923) | [Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) |
| **HiCache** | 2025 | DiT | Predictive (Hermite) | ~2.6× | [2508.16984](https://arxiv.org/abs/2508.16984) | - |
| **FoCa** | 2025 | DiT | Predictive (BDF2+Heun) | ~2.5× | - | - |
| **AB-Cache** | 2025 | DiT | Predictive (Adams-Bashforth) | ~2.3× | - | - |
| **FEB-Cache** | 2025 | DiT | Frequency (Attn/MLP 分频) | ~2.0× | [2503.07120](https://arxiv.org/abs/2503.07120) | - |
| **FreqCa** | 2025 | DiT | Frequency (低频复用+高频预测) | ~7.14× | [2510.08669](https://arxiv.org/abs/2510.08669) | - |
| **SeaCache** | CVPR 2026 | DiT | Spectral-Evolution-Aware | ~2.5× | - | [jiwoogit/SeaCache](https://github.com/jiwoogit/SeaCache) |
| **🔥 SpectralCache** | 2026 | DiT (FLUX/PixArt) | Hybrid (TADS×CEB×FDC) | **2.46×** | Coming soon | [leeguandong/SpectralCache](https://github.com/leeguandong/SpectralCache) |
| **🔥 LayerCache** | 2026 | Flow Matching (Qwen-Image/FLUX) | Layer-Adaptive + JVP | **1.71×** | Coming soon | [leeguandong/LayerCache](https://github.com/leeguandong/LayerCache) |
| **MixCache** | 2025 | Video DiT | Mixture-of-Cache | ~2.2× | [2508.12691](https://arxiv.org/abs/2508.12691) | - |
| **BWCache** | 2025 | Video DiT | Block-Wise | ~2.0× | [2509.13789](https://arxiv.org/abs/2509.13789) | - |

> 备注：加速比对应各自论文的最佳无损/近无损配置，数值来自原论文的 FLUX、SD3、PixArt、CogVideoX、Open-Sora 等主流 backbone。

### 1.2 演化时间线

```
2023Q4  DeepCache                      (UNet feature 复用开创)
2024Q1  T-GATE                          (cross-attn freeze)
2024Q2  FORA / Δ-DiT                    (UNet cache 思路迁移到 DiT)
2024Q3  PAB                             (视频 DiT 金字塔广播)
2024Q4  TeaCache / ToCa / HarmoniCa    (timestep 阈值 / token 级 / learning)
2025Q1  TaylorSeer                     (Cache-then-Forecast 开创)
2025Q2  MagCache / Chipmunk / LazyDiT  (幅值定律 / 稀疏增量 / learned skip)
2025Q3  HiCache / FoCa / AB-Cache      (Hermite / ODE 数值积分)
2025Q4  FreqCa / FEB-Cache / DiCache   (频域 / 自触发)
2026Q1  SeaCache / SpectralCache / LayerCache  (频谱演化 / 频域 hybrid / 层异质 + JVP)
```

## 2. 按缓存粒度分类（What is cached）

本节从 **"到底在缓存什么对象"** 的角度做一次正交切分，和 §3 的 "调度策略" 配合使用。每个方法通常在一个主要粒度上做文章，少数混合方法会跨多个粒度（见 §2.9 矩阵）。

### 2.1 Step Cache（整步输出）

**缓存对象**：整个 transformer / UNet 在某个 timestep 的输出（或 residual）→ 下一步或下几步直接复用。这是最粗的粒度，也是最主流的做法。

| 方法 | 缓存什么 | 复用方式 |
|------|---------|----------|
| **DeepCache** | UNet 深层 feature | 固定间隔复用 |
| **TeaCache** | 整步 residual | 基于 timestep embedding 阈值决定复用 |
| **FBCache** | 首个 block 的 residual 作触发，整步跳过 | 阈值触发 |
| **TaylorSeer** | 历史多步的 step 输出 | Taylor 外推预测当前步 |
| **MagCache** | 整步 residual | 几何衰减幅值律预测 |
| **EasyCache** | transformation vector | runtime self-correct |
| **FasterCache** | 整步 feature + CFG 分支 | 混合复用 |

### 2.2 Block Cache（Transformer Block 输出）

**缓存对象**：单个或连续几个 transformer block 的输出。介于 step 和 layer 之间的粒度。

| 方法 | 缓存什么 |
|------|---------|
| **Δ-DiT** | 各 block 的 residual 增量 |
| **DBCache** | 中段 block 群（Probe-Main-Corrector 三段划分）|
| **BWCache** | 视频 DiT 的 block-wise 输出 |
| **Skip-DiT** | 深层 block 的 long-skip 路径 |
| **Cache Me if You Can** | 每个 block 独立阈值 |
| **HarmoniCa** | block 级（learning-based 调度）|
| **LayerCache** (本作) | 层组（Shallow/Middle/Deep）级输出 + JVP |

### 2.3 Attention Cache（注意力模块）

**缓存对象**：attention 的输出、attention map、或 KV。基于 "attention 冗余度比 MLP 更高" 的观察。

| 方法 | 缓存什么 |
|------|---------|
| **T-GATE** | cross-attention 输出（收敛点后 freeze）|
| **PAB** | self/cross/temporal attention 各自按类型广播 |
| **FEB-Cache** (Attn 分支) | 后期阶段的 attention 输出（低频结构）|
| **FasterCache** (attention 部分) | attention feature 跨步复用 |

### 2.4 MLP / FFN Cache

**缓存对象**：transformer 中 MLP / FFN 模块的输出。

| 方法 | 缓存什么 |
|------|---------|
| **FEB-Cache** (MLP 分支) | 早期阶段的 MLP 输出（高频细节）|
| **FORA** (MLP 部分) | MLP 在固定区间内复用 |

### 2.5 Token Cache（token 子集）

**缓存对象**：每个 block 内一部分 token 的激活，激活态 / 静态 token 区分处理。

| 方法 | 缓存什么 |
|------|---------|
| **ToCa** | 低敏感度 token 的每层激活 |
| **DuCa** | token × layer 双层 |
| **FastCache** | 静态 token 用学习的线性近似映射 |
| **Chipmunk** | 低贡献 activation 的 column-sparse cache |

### 2.6 Frequency-Band Cache（频带分解）

**缓存对象**：对特征做频域分解后的低频 / 高频分量，分别缓存。

| 方法 | 缓存什么 |
|------|---------|
| **FEB-Cache** | Attn 偏低频 / MLP 偏高频，分阶段切换对象 |
| **FreqCa** | 低频 reuse + 高频用二阶 Hermite 外推，CRF 残差降内存 99% |
| **SeaCache** | 跟踪频谱演化触发刷新 |
| **SpectralCache** (本作) | 低频 γ=0.8 严 / 高频 γ=1.5 松 的非对称阈值 |

### 2.7 CFG-Branch Cache

**缓存对象**：Classifier-Free Guidance 中 unconditional / conditional 分支的输出。

| 方法 | 缓存什么 |
|------|---------|
| **CFG-Cache** (FasterCache 子模块) | uncond 分支跨步复用 |
| **FasterCache** (CFG 频域分解) | 把 CFG 差异按频域分开缓存 |

### 2.8 Residual Cache（层间残差）

**缓存对象**：层与层之间的残差、或速度场的导数估计。

| 方法 | 缓存什么 |
|------|---------|
| **Δ-DiT** | block 级 residual 增量 |
| **Chipmunk** | activation 级 residual |
| **LayerCache** (本作) | JVP（Jacobian-Vector Product）形式的速度残差，用 MeanFlow Identity 外推 |
| **AB-Cache / FoCa / HiCache** | 把 cache 看作 ODE 数值积分的状态量 |

### 2.9 缓存粒度 × 调度策略 交叉矩阵

列 = §3 的调度策略；行 = §2 的缓存粒度。◆ = 主要归属，○ = 次要命中。

| 粒度 \ 策略 | Static | Timestep-Adaptive | Layer-Adaptive | Predictive | Token-Level | Frequency-Aware | CFG | Hybrid |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Step Cache**       | DeepCache / FORA ◆ | TeaCache / FBCache / MagCache / EasyCache ◆ | — | TaylorSeer / HiCache / AB-Cache / FoCa ◆ | — | — | — | FasterCache ○ |
| **Block Cache**      | Δ-DiT ◆ | Cache Me if You Can ◆ | DBCache / Skip-DiT / HarmoniCa / **LayerCache** ◆ | — | — | — | — | BWCache ○ |
| **Attention Cache**  | T-GATE ◆ | — | — | — | — | FEB-Cache (Attn) ○ | — | PAB ◆ / FasterCache ○ |
| **MLP Cache**        | FORA (MLP) ◆ | — | — | — | — | FEB-Cache (MLP) ◆ | — | — |
| **Token Cache**      | — | Chipmunk ◆ | — | — | ToCa / DuCa / FastCache ◆ | — | — | — |
| **Frequency Band**   | — | — | — | FreqCa (高频预测) ○ | — | FreqCa / SeaCache / FEB-Cache / **SpectralCache** ◆ | — | **SpectralCache** ○ |
| **CFG Branch**       | — | — | — | — | — | FasterCache (CFG+freq) ○ | CFG-Cache ◆ | — |
| **Residual**         | Δ-DiT ◆ | Chipmunk ○ | **LayerCache** (JVP) ◆ | AB-Cache / FoCa / HiCache ◆ | — | — | — | — |

> **怎么读这张表**：
> - 横向看：一个调度策略下都有哪些缓存粒度的代表。
> - 纵向看：同一粒度下不同调度思路的演化。
> - **LayerCache** 同时命中 *Block / Residual* 粒度 + *Layer-Adaptive / Predictive* 策略（所以在 Hybrid 意义上是"层粒度 + 预测"）。
> - **SpectralCache** 同时命中 *Frequency Band* 粒度 + *Frequency-Aware / Hybrid* 策略。

## 3. 按调度策略详述（How to decide）

### 3.1 Static Caching（固定调度）

固定复用区间 / 固定层集合，**无运行时决策**。

* **DeepCache**：
  * 地址：https://github.com/horseee/DeepCache ![](https://img.shields.io/github/stars/horseee/DeepCache.svg)
  * 论文：[CVPR 2024](https://arxiv.org/abs/2312.00858)
  * 简介：首个系统化利用扩散模型时序冗余的 training-free 方法。基于 UNet skip connection 观察：高层特征跨相邻步变化平缓。DeepCache 跨步复用 UNet 上采样路径的 deep feature，浅层每步重算。在 SD 1.5/2.1 上可取得约 2.3× 加速，几乎无损。**仅适用于 UNet，DiT 不适用**。

* **FasterDiffusion**：
  * 简介：发现 UNet encoder 对相邻 step 的输出非常相似，提出 encoder propagation：跨步复用 encoder 输出，仅 decoder 继续更新。同时引入 parallel decoding 降低串行开销。

* **T-GATE**：
  * 地址：https://github.com/HaozheLiu-ST/T-GATE ![](https://img.shields.io/github/stars/HaozheLiu-ST/T-GATE.svg)
  * 论文：[arXiv 2404.02747](https://arxiv.org/abs/2404.02747)
  * 简介：发现 cross-attention 在早期去噪阶段即收敛，之后几乎不变。T-GATE 在收敛点直接 freeze cross-attention 输出并跨步复用。适用 SD、PixArt、LCM。V2 进一步支持 DiT。

* **FORA** (First-Order Residual Approximation)：
  * 论文：[arXiv 2407.01425](https://arxiv.org/abs/2407.01425)
  * 简介：把 DeepCache 思路迁移到 DiT，固定间隔复用 self-attn / MLP 输出。是 DiT cache 领域最早的 baseline 之一。

* **Δ-DiT**：
  * 论文：[arXiv 2406.01125](https://arxiv.org/abs/2406.01125)
  * 简介：缓存 residual 增量而非绝对值，并根据生成阶段（布局 / 细节）动态调整不同 block 的缓存侧重。

* **PAB (Pyramid Attention Broadcast)**：
  * 地址：https://github.com/NUS-HPC-AI-Lab/VideoSys ![](https://img.shields.io/github/stars/NUS-HPC-AI-Lab/VideoSys.svg)
  * 论文：[ICLR 2025 / arXiv 2408.12588](https://arxiv.org/abs/2408.12588)
  * 简介：视频 DiT 的 attention 差分呈 U 形，空间 / 时间 / cross attention 稳定性不同。PAB 按注意力类型设置不同广播半径（金字塔式），在 Open-Sora / Latte / Open-Sora-Plan 上达到 21.6 FPS 实时生成，10.6× 加速。

### 3.2 Timestep-Adaptive（时步自适应）

通过阈值 / 相似度 / 误差预算决定**当前步是否重算**，是 DiT cache 的主流。

* **TeaCache**：
  * 地址：https://github.com/ali-vilab/TeaCache ![](https://img.shields.io/github/stars/ali-vilab/TeaCache.svg)
  * 论文：[CVPR 2025 / arXiv 2411.19108](https://arxiv.org/abs/2411.19108)
  * 简介：当前应用最广泛的 baseline。用 **timestep embedding** 的 L1 距离作为变化估计，累计超阈值才刷新。接入 CogVideoX / HunyuanVideo / Wan / FLUX / Mochi 等主流模型生态。

* **FBCache (First-Block Cache)**：
  * 地址：https://github.com/chengzeyi/ParaAttention ![](https://img.shields.io/github/stars/chengzeyi/ParaAttention.svg)
  * 简介：只用第一个 transformer block 的 residual 作为触发信号，实现简单、开销低。常作为 TeaCache 的轻量工程替代。

* **MagCache**：
  * 地址：https://github.com/Zehong-Ma/MagCache ![](https://img.shields.io/github/stars/Zehong-Ma/MagCache.svg)
  * 论文：[arXiv 2506.09045](https://arxiv.org/abs/2506.09045)
  * 简介：把 residual 演化建模为几何衰减，提出统一幅值律，无需 calibration 即可 plug-and-play。

* **EasyCache**：
  * 论文：[ICCV 2025 / arXiv 2507.02860](https://arxiv.org/abs/2507.02860)
  * 简介：runtime adaptive self-correct：相对变换率 + 累计偏差双指标，自适应调整阈值，2.1–3.3× 加速。

* **LazyDiT**：
  * 论文：[AAAI 2025 / arXiv 2412.12444](https://arxiv.org/abs/2412.12444)
  * 简介：在每个 transformer layer 前插入线性预测器，用一阶 Taylor 近似 learn 相似度，决定是否跳过该层计算。

* **Chipmunk**：
  * 论文：[arXiv 2506.03275](https://arxiv.org/abs/2506.03275)
  * 简介：发现 5–25% 的 activation 占 70–90% 变化量，提出 column-sparse activation cache，硬件友好。

* **Cache Me if You Can (Block Cache)**：
  * 论文：[CVPR 2024 / arXiv 2312.03209](https://arxiv.org/abs/2312.03209)
  * 简介：每个 block 有独立阈值，相对变化超阈值才刷新，早期的 block-wise 阈值工作。

### 3.3 Layer-Adaptive（深度自适应）

在**层深度维度**决定哪些层算 / 哪些层缓存，代表了"不同层对 cache 敏感度不同"的洞察。

* **HarmoniCa**：
  * 论文：[arXiv 2410.01723](https://arxiv.org/abs/2410.01723)
  * 简介：首个 learning-based cache schedule 工作。在完整 denoising trajectory 上训练 cache controller，解决 train-inference mismatch。

* **AdaCache**：
  * 地址：https://adacache-dit.github.io/
  * 论文：[arXiv 2411.02397](https://arxiv.org/abs/2411.02397)
  * 简介：视频 DiT 的 content-adaptive schedule——**每个 video 都有独立的 cache 计划**。结合 residual 变化 + motion regularization，在 Open-Sora 上达到 4.49× 加速。

* **DBCache (Dual Block Cache)**：
  * 简介：DiT block stack 分三段：**Probe（前段全算）→ Main（中段阈值缓存）→ Corrector（尾段纠正）**。典型的概率-决策-纠错架构。

* **Skip-DiT**：
  * 论文：[ICCV 2025 / arXiv 2411.17616](https://arxiv.org/abs/2411.17616)
  * 简介：借鉴 long-skip-connection 思想，深层做 static cache，浅层每步 update，解决深层 DiT 稳定性问题。

* **🔥 LayerCache (CVPR 2026)**：
  * 地址：https://github.com/UnicomAI/LayerCache
  * 简介：发现 flow matching 模型中 transformer 的**层组速度异质性**——Shallow / Middle / Deep 有不同的稳定度：浅层稳定可激进缓存（98%），中层中等（52%），深层高度易变（0% 缓存）。提出 **3D schedule (timestep × layer group × JVP span K)** + greedy budget allocation + JVP-based forecasting。在 Qwen-Image 上 1.71× 加速，PSNR 34.16，显著优于 MeanCache baseline。

### 3.4 Predictive / Cache-then-Forecast（预测类）

把 cache 升级为**数值积分 / 多项式外推**：用历史 step 的特征预测未来 step。

* **TaylorSeer**：
  * 地址：https://github.com/Shenyi-Z/TaylorSeer ![](https://img.shields.io/github/stars/Shenyi-Z/TaylorSeer.svg)
  * 论文：[ICCV 2025 / arXiv 2503.06923](https://arxiv.org/abs/2503.06923)
  * 简介：**Cache-then-Forecast 范式开创**。用多步历史特征做差分近似各阶导数，Taylor 级数外推未来 step 的特征。奠定了后续预测类方法的理论基础。

* **HiCache**：
  * 论文：[arXiv 2508.16984](https://arxiv.org/abs/2508.16984)
  * 简介：发现 DiT 特征导数的近似呈多元高斯特征，改用 **Hermite 多项式**（高斯共轭的理论最优基）替换 Taylor 基，plug-and-play，显著提升稳定性。

* **FoCa**：
  * 简介：两阶段：**BDF2 预测 + Heun 校正**，把 cache 直接建模为 feature ODE 的数值积分。

* **AB-Cache**：
  * 简介：**Adams-Bashforth** 多步法，解释了 U 形相似度现象的数学根源——相邻 step 输出之间的线性关系。

### 3.5 Token-Level / Granularity（细粒度）

在 **token 维度**决定哪些 token 激活 / 哪些 token 用旧值。

* **ToCa (Token-wise Feature Caching)**：
  * 地址：https://github.com/Shenyi-Z/ToCa ![](https://img.shields.io/github/stars/Shenyi-Z/ToCa.svg)
  * 论文：[ICLR 2025 / arXiv 2410.05317](https://arxiv.org/abs/2410.05317)
  * 简介：首次在 **token 粒度**研究 DiT cache。发现不同 token 对缓存敏感度显著不同，细粒度选择适合 cache 的 token。

* **DuCa (Dual Feature Cache)**：
  * 论文：[arXiv 2412.18911](https://arxiv.org/abs/2412.18911)
  * 简介：ToCa 升级，token × layer 双层缓存。

* **FastCache**：
  * 论文：[arXiv 2505.20353](https://arxiv.org/abs/2505.20353)
  * 简介：静态 token 用**可学习线性近似**直接映射，活跃 token 全算，可达 4.5× 激进加速。

* **DiCache**：
  * 论文：[arXiv 2508.17356](https://arxiv.org/abs/2508.17356)
  * 简介：**让模型自己决定 cache**——用 shallow feature 作为 probe，基于变化触发重算。

### 3.6 Frequency-Aware（频域类）

在**频率维度**区分高低频特征的不同时序行为。

* **FEB-Cache**：
  * 论文：[arXiv 2503.07120](https://arxiv.org/abs/2503.07120)
  * 简介：发现 **Attention 偏低频结构、MLP 偏高频细节**的互补频谱敏感性。提出分阶段频域缓存表：早期重 MLP cache，后期重 Attention cache。

* **FreqCa**：
  * 论文：[arXiv 2510.08669](https://arxiv.org/abs/2510.08669)
  * 简介：低频**相似度高但连续性差** → 直接复用；高频**连续性高但相似度差** → 二阶 Hermite 外推。**CRF (Cumulative Residual Feature)** 把 cache 内存降 99%，可达 7.14× 加速。

* **SeaCache (Spectral-Evolution-Aware Cache)**：
  * 地址：https://github.com/jiwoogit/SeaCache
  * 论文：CVPR 2026
  * 简介：跟踪**频谱演化**触发刷新，频域视角的动态 cache。

* **🔥 SpectralCache (本作)**：
  * 地址：https://github.com/leeguandong/SpectralCache
  * 简介：三轴正交的 Hybrid 频域 cache：
    - **TADS** (Timestep-Aware Dynamic Scheduling)：cosine bell 时步阈值调度
    - **CEB** (Cumulative Error Budget)：连续缓存上限 C_max，防误差级联
    - **FDC** (Frequency-Decomposed Caching)：高低频带**非对称阈值**（低频严 γ=0.8 / 高频松 γ=1.5）
  * 在 FLUX.1-schnell 上 **2.46× 加速**，比 TeaCache 快 16%。

### 3.7 CFG-Level Caching

针对 **CFG 分支**（conditional / unconditional）的冗余做缓存。

* **CFG-Cache** (FasterCache 子模块)：
  * 简介：cond / uncond 分支输出非常相似，可跨步复用 uncond 分支。

* **FasterCache 的 CFG 频域分解**：
  * 论文：[ICLR 2025 / arXiv 2410.19355](https://arxiv.org/abs/2410.19355)
  * 简介：把 CFG 差异分解为高低频两部分，分开做 cache 决策。

### 3.8 Video DiT Cache（视频专用）

视频 DiT 具有额外的时间维度冗余，往往配合更激进的 cache 策略。

* **PAB** → 见 3.1
* **FasterCache** → 见 3.9
* **AdaCache** → 见 3.3
* **MixCache** (Mixture-of-Cache)：
  * 论文：[arXiv 2508.12691](https://arxiv.org/abs/2508.12691)
  * 简介：多个 cache 策略组成 mixture，router 动态选择。
* **BWCache** (Block-Wise Cache)：
  * 论文：[arXiv 2509.13789](https://arxiv.org/abs/2509.13789)
  * 简介：视频 DiT 的 block-wise 缓存。
* **EasyCache** → 见 3.2

### 3.9 Hybrid / Multi-Dimensional（混合类）

组合多个轴（time × layer × frequency × CFG × token）的混合方法。

* **FasterCache**：
  * 地址：https://github.com/Vchitect/FasterCache ![](https://img.shields.io/github/stars/Vchitect/FasterCache.svg)
  * 论文：[ICLR 2025 / arXiv 2410.19355](https://arxiv.org/abs/2410.19355)
  * 简介：Feature × CFG × Frequency 三轴融合，Vchitect-2.0 上 1.67×。
* **SpectralCache** → 见 3.6
* **LayerCache** → 见 3.3（Layer + Predictive 两轴）

## 4. 测评

### 4.1 常用评测指标

| 类别 | 指标 | 含义 |
|------|------|------|
| 加速 | **Speedup** | 相对无 cache 的 wall-clock 加速比 |
| 加速 | **Latency / step** | 单步推理时延 |
| 像素级 | **PSNR ↑** | 峰值信噪比（vs. 无 cache 输出）|
| 结构级 | **SSIM ↑** | 结构相似度 |
| 感知级 | **LPIPS ↓** | 感知距离（AlexNet/VGG）|
| 分布级 | **FID ↓** | Frechet Inception Distance |
| 文图对齐 | **CLIP-Score ↑** | text-image 对齐 |
| 视频时序 | **VBench** | 视频质量多维度评测 |
| 视频时序 | **Temporal Flickering / Motion Smoothness** | 时序连贯性 |

### 4.2 基线模型

**图像**：SD 1.5 / SDXL / PixArt-α / PixArt-Σ / FLUX.1-dev / FLUX.1-schnell / SD3 / Qwen-Image / Z-Image / LongCat-Image

**视频**：Open-Sora / Open-Sora-Plan / Latte / CogVideoX / HunyuanVideo / Wan2.1 / Wan2.2 / Mochi / Vchitect-2.0

### 4.3 常用 Benchmark

| Benchmark | 用途 | 链接 |
|-----------|------|------|
| **COCO-30K** | 图像 FID / CLIP 评测 | - |
| **MJHQ-30K** | 图像质量评测 | - |
| **GenEval** | 图像文图对齐评测 | [djghosh13/geneval](https://github.com/djghosh13/geneval) |
| **DPG-Bench** | 长文图对齐 | [TencentQQGYLab/ELLA](https://github.com/TencentQQGYLab/ELLA) |
| **VBench** | 视频生成 16 维度评测 | [Vchitect/VBench](https://github.com/Vchitect/VBench) |
| **OneIG-Bench** | 统一图像生成评测 | - |

## 5. 工程与工具

| 工具 | 说明 |
|------|------|
| **xFuser / ParaAttention** | 并行 + cache 一体化框架，TeaCache / FBCache / SpectralCache 的集成入口 |
| **VideoSys** | 视频 DiT 推理优化框架，PAB 官方实现载体 |
| **Diffusers pipeline hooks** | 通过 hook 注入 cache 的通用模式 |
| **vLLM-Omni Diffusion Cache** | vLLM 引入的 diffusion cache 工程化实现 |
| **TensorRT-LLM / TensorRT** | cache + low-precision 联合部署 |

## 6. 相关综述

* **A Survey on Cache Methods in Diffusion Models: Toward Efficient Multi-Modal Generation** ([arXiv 2510.19755](https://arxiv.org/abs/2510.19755)) — 2025 年 10 月，目前最全最新的 cache 综述，将方法分为 static / timestep-adaptive / layer-adaptive / predictive / hybrid 五大类。
* **Efficient Diffusion Models: A Comprehensive Survey** — 覆盖量化、蒸馏、cache、并行等全方向加速。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=leeguandong/Awesome-Dit-Cache&type=Date)](https://star-history.com/#leeguandong/Awesome-Dit-Cache&Date)

## License

Apache License 2.0
