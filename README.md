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

> A curated list of feature caching / reuse-based acceleration methods for Diffusion Models and Diffusion Transformers (DiT), covering image, video, audio, and flow-matching generators.

本项目旨在收集和梳理 **扩散模型（UNet / DiT / Flow Matching）推理加速** 中的 **Cache / Reuse 类方法**——包括 Timestep-Adaptive、Layer-Adaptive、Predictive (Cache-then-Forecast)、Fine-Grained (Token / Region / Channel)、Frequency-Aware、CFG-Level、Video-DiT 与 Service/System-Level 等主流范式。以训练无关（training-free）的特征缓存与近邻近似为主，也覆盖以 cache 为核心的轻量校准和系统/硬件协同。

如果本项目能给您带来一点点帮助，麻烦点个⭐️吧～

同时也欢迎大家贡献本项目未收录的论文、开源实现。提供新的仓库信息请发起 PR，并按照本项目的格式提供仓库链接、arXiv 编号、会议、简介等信息，感谢～

## About

**Why this repo / 为什么做这个仓库**

过去两年里，Diffusion / DiT 推理加速领域的 Cache 类方法井喷式出现——从 2023 年 DeepCache 把"特征复用"第一次系统化，到 2025 年 TaylorSeer、HiCache、FoCa 把它升级成"数值积分式预测"，再到 2026 年的 Spectrum（Chebyshev 全局近似）、JiT（空间 ODE 稀疏，7×）、MeanCache（JVP 平均速度）、SenCache（敏感度驱动）、MoECa（MoE 分支级缓存）、LearniBridge（LoRA 校准）、SyncCache（音频驱动人像 cache），以及 Q3 新出现的 ACID（临界步双阈值）、Kaleido（通道级局部结果复用）、FlashDiff（区域级服务调度）、CODA / DSTAR（compute-cache 与时空冗余的软硬协同）、DiTango（并行 attention state 复用）、OmniCache（四级分层复用）、FeatFix / RACER（把"验证"变成"校正"、用预测分歧度做闭环控制）、OnlineCache（policy-gradient 学调度）、EchoCache / WorldDynCache（音频跨模态与 world model）、HeadCast / EVO / CachedSearch（AR head 级 KV 通路、diffusion policy 进化调度、cache × test-time search）等，把预测基底、空间冗余、MoE 架构适配、轻量校准、模态异质性、误差闭环控制、系统协同与**算力预算分配**等新维度一并拉入。方法演化已经跨越了 **"复用 → 调度 → 预测 → 多轴混合 → 架构/空间/频域/模态/系统自适应 → 误差闭环与预算分配"** 六个阶段。但这个方向缺少一个**统一的中文索引**：论文分散在 CVPR / ICLR / ICCV / NeurIPS / ACM MM / arXiv，Video DiT 与 Image DiT 的工作被割裂收录，新老 baseline 对比困难。

这个仓库就是为了填这个缺口：
- **以"调度策略"为主轴**（Static → Timestep → Layer → Predictive → Fine-Grained → Frequency → CFG → Hybrid），把 2023–2026 的代表性方法一次性摆到同一张表里。
- **双语**介绍，尽量补齐论文、arXiv、项目页与代码状态，方便检索与追溯。
- **覆盖范围** = Image DiT（FLUX / SD3 / PixArt / Qwen-Image）+ Video DiT（CogVideoX / HunyuanVideo / Wan / Open-Sora）+ Audio Diffusion（Stable Audio Open）+ Flow Matching，不只是狭义的 image 场景。

**Scope / 收录边界**

✅ 收录：以 **training-free** 特征缓存、激活/局部结果复用为主；覆盖时步、层、token、region、channel、频域、CFG 与跨请求维度，也收录以 cache 为核心的轻量预测器/校准器，以及直接支撑 cache 的服务系统和软硬协同工作。
❌ 暂不收录：不以 cache / reuse 为核心的量化（FP8/INT8）、纯蒸馏（step distillation / consistency models）、纯注意力内核优化（SageAttn、FlashAttn）、纯并行推理（xDiT / USP）——这些虽然常与 cache 叠加使用，但各自有独立的加速机理，另起仓库更合适。

**Maintainer**

由 [@leeguandong](https://github.com/leeguandong) 维护，欢迎 issue / PR。如果你在做 cache 相关工作希望被收录，请提供：论文 arXiv、代码仓库、目标模型、加速比、**两轴归属**（缓存 / 复用粒度 §2 + 调度策略 §3）。

## 目录

- [About](#about)
- [1. 方法汇总](#1-方法汇总)
  - [1.1 方法全景表](#11-方法全景表)
  - [1.2 演化时间线](#12-演化时间线)
- [2. 按缓存 / 复用粒度分类（What is cached or reused）](#2-按缓存--复用粒度分类what-is-cached-or-reused)
  - [2.1 Step Cache（整步输出）](#21-step-cache整步输出)
  - [2.2 Block Cache（Transformer Block 输出）](#22-block-cachetransformer-block-输出)
  - [2.3 Attention Cache（注意力模块）](#23-attention-cache注意力模块)
  - [2.4 MLP / FFN Cache](#24-mlp--ffn-cache)
  - [2.5 Fine-Grained Cache（token / region / channel）](#25-fine-grained-cachetoken--region--channel)
  - [2.6 Frequency-Band Cache（频带分解）](#26-frequency-band-cache频带分解)
  - [2.7 CFG-Branch Cache](#27-cfg-branch-cache)
  - [2.8 Residual Cache（层间残差）](#28-residual-cache层间残差)
  - [2.9 缓存 / 复用粒度 × 调度策略 交叉矩阵](#29-缓存--复用粒度--调度策略-交叉矩阵)
- [3. 按调度策略详述（How to decide）](#3-按调度策略详述how-to-decide)
  - [3.1 Static Caching（固定调度）](#31-static-caching固定调度)
  - [3.2 Timestep-Adaptive（时步自适应）](#32-timestep-adaptive时步自适应)
  - [3.3 Layer-Adaptive（深度自适应）](#33-layer-adaptive深度自适应)
  - [3.4 Predictive / Cache-then-Forecast（预测类）](#34-predictive--cache-then-forecast预测类)
  - [3.5 Fine-Grained / Granularity（token / region / channel）](#35-fine-grained--granularitytoken--region--channel)
  - [3.6 Frequency-Aware（频域类）](#36-frequency-aware频域类)
  - [3.7 CFG-Level Caching](#37-cfg-level-caching)
  - [3.8 Video DiT Cache（视频专用）](#38-video-dit-cache视频专用)
  - [3.9 Hybrid / Multi-Dimensional（混合类）](#39-hybrid--multi-dimensional混合类)
  - [3.10 Service-Level Cache（跨请求 / 区域调度）](#310-service-level-cache跨请求--区域调度)
- [4. 测评](#4-测评)
  - [4.1 常用评测指标](#41-常用评测指标)
  - [4.2 基线模型](#42-基线模型)
  - [4.3 常用 Benchmark](#43-常用-benchmark)
- [5. 工程、系统与硬件](#5-工程系统与硬件)
- [6. 相关综述](#6-相关综述)
- [Star History](#star-history)
- [License](#license)

## 1. 方法汇总

### 1.1 方法全景表

| 方法 | 会议/年 | 目标模型 | 范式 | 典型加速 | arXiv | 代码 |
|------|--------|---------|------|---------|-------|------|
| **DeepCache** | CVPR 2024 | UNet (SD 1.5/2.x) | Static + 时步 | ~2.3× | [2312.00858](https://arxiv.org/abs/2312.00858) | [horseee/DeepCache](https://github.com/horseee/DeepCache) |
| **FasterDiffusion** | NeurIPS 2024 | UNet | Static (encoder skip) | ~1.8× | [2312.09608](https://arxiv.org/abs/2312.09608) | [hutaiHang/Faster-Diffusion](https://github.com/hutaiHang/Faster-Diffusion) |
| **T-GATE V1/V2** | 2024 | SD / PixArt / LCM | 阶段式 (cross-attn freeze) | ~1.5× | [2404.02747](https://arxiv.org/abs/2404.02747) | [HaozheLiu-ST/T-GATE](https://github.com/HaozheLiu-ST/T-GATE) |
| **FORA** | 2024 | DiT | Static 固定区间 | ~1.8× | [2407.01425](https://arxiv.org/abs/2407.01425) | [prathebaselva/FORA](https://github.com/prathebaselva/FORA) |
| **Δ-DiT** | 2024 | DiT | Static (residual cache) | ~1.6× | [2406.01125](https://arxiv.org/abs/2406.01125) | - |
| **Block Cache / Cache Me if You Can** | CVPR 2024 | UNet | Layer-Adaptive 阈值 | ~1.8× | [2312.03209](https://arxiv.org/abs/2312.03209) | - |
| **BlockDance** | CVPR 2025 | DiT / Video DiT | Block-level (STSS block) + 时步 | 1.25–1.50× | [2503.15927](https://arxiv.org/abs/2503.15927) | - |
| **PAB** (Pyramid Attention Broadcast) | ICLR 2025 | Video DiT (Open-Sora/Latte) | Static × attention 类型 | ~10.6× FPS | [2408.12588](https://arxiv.org/abs/2408.12588) | [NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys) |
| **FasterCache** | ICLR 2025 | Video DiT | Hybrid (feature + CFG) | ~1.67× | [2410.19355](https://arxiv.org/abs/2410.19355) | [Vchitect/FasterCache](https://github.com/Vchitect/FasterCache) |
| **AdaCache** | 2024 | Video DiT | Layer-Adaptive (per-video) | ~4.49× | [2411.02397](https://arxiv.org/abs/2411.02397) | [adacache-dit](https://adacache-dit.github.io/) |
| **ProfilingDiT** | ICCV 2025 | Video DiT (Wan2.1) | Block profiling (FG/BG 解耦) | ~2.01× | [2504.03140](https://arxiv.org/abs/2504.03140) | [GeekGuru123/ProfilingDiT](https://github.com/GeekGuru123/ProfilingDiT) |
| **TeaCache** | CVPR 2025 | DiT / Video DiT | Timestep-Adaptive (阈值) | ~2.1× | [2411.19108](https://arxiv.org/abs/2411.19108) | [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) |
| **FBCache** (First-Block Cache) | 2024 | DiT | Timestep-Adaptive (首层触发) | ~1.87× | - | [chengzeyi/ParaAttention](https://github.com/chengzeyi/ParaAttention) |
| **HarmoniCa** | 2024 | UNet | Layer-Adaptive (learning) | ~1.7× | [2410.01723](https://arxiv.org/abs/2410.01723) | [ModelTC/HarmoniCa](https://github.com/ModelTC/HarmoniCa) |
| **MagCache** | 2025 | DiT | Timestep (幅值定律) | ~2.0× | [2506.09045](https://arxiv.org/abs/2506.09045) | [Zehong-Ma/MagCache](https://github.com/Zehong-Ma/MagCache) |
| **EasyCache** | ICCV 2025 | Video DiT | Timestep (runtime self-correct) | 2.1–3.3× | [2507.02860](https://arxiv.org/abs/2507.02860) | [H-EmbodVis/EasyCache](https://github.com/H-EmbodVis/EasyCache) |
| **LazyDiT** | AAAI 2025 | DiT | Timestep (learned skip) | ~1.9× | [2412.12444](https://arxiv.org/abs/2412.12444) | [shawnricecake/lazydit](https://github.com/shawnricecake/lazydit) |
| **Chipmunk** | 2025 | DiT | Timestep (稀疏增量) | ~2.5× | [2506.03275](https://arxiv.org/abs/2506.03275) | [sandyresearch/chipmunk](https://github.com/sandyresearch/chipmunk) |
| **ToCa** | ICLR 2025 | DiT | Token-Level | ~1.5× | [2410.05317](https://arxiv.org/abs/2410.05317) | [Shenyi-Z/ToCa](https://github.com/Shenyi-Z/ToCa) |
| **DuCa** (Dual Feature Cache) | 2024 | DiT | Token × Layer 双层 | ~1.9× | [2412.18911](https://arxiv.org/abs/2412.18911) | [Shenyi-Z/DuCa](https://github.com/Shenyi-Z/DuCa) |
| **FastCache** | 2025 | DiT | Token 线性近似 | ~4.5× | [2505.20353](https://arxiv.org/abs/2505.20353) | [NoakLiu/FastCache-xDiT](https://github.com/NoakLiu/FastCache-xDiT) |
| **DiCache** | 2025 | DiT | shallow probe 自触发 | ~2.3× | [2508.17356](https://arxiv.org/abs/2508.17356) | [Bujiazi/DiCache](https://github.com/Bujiazi/DiCache) |
| **ClusCa** | ACM MM 2025 | DiT (FLUX/HunyuanVideo) | Token Cluster (KMeans 簇内传播) | **4.96× FLUX** | [2509.10312](https://arxiv.org/abs/2509.10312) | [Shenyi-Z/Cache4Diffusion](https://github.com/Shenyi-Z/Cache4Diffusion) |
| **DBCache** | 2025 | DiT | Probe-Decide-Correct | ~2.0× | - | [vipshop/cache-dit](https://github.com/vipshop/cache-dit) |
| **Skip-DiT** | ICCV 2025 | DiT | Long-Skip-Connection + cache | 1.5–2× | [2411.17616](https://arxiv.org/abs/2411.17616) | [OpenSparseLLMs/Skip-DiT](https://github.com/OpenSparseLLMs/Skip-DiT) |
| **TaylorSeer** | ICCV 2025 | DiT | Predictive (Taylor) | ~2.4× | [2503.06923](https://arxiv.org/abs/2503.06923) | [Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) |
| **HiCache** | 2025 | DiT | Predictive (Hermite) | ~2.6× | [2508.16984](https://arxiv.org/abs/2508.16984) | [fenglang918/HiCache](https://github.com/fenglang918/HiCache) |
| **FoCa** | 2025 | DiT / Video DiT | Predictive (Feature-ODE) | **5.50× FLUX** | [2508.16211](https://arxiv.org/abs/2508.16211) | - |
| **AB-Cache** | 2025 | DiT / Video DiT | Predictive (Adams-Bashforth) | ~3× | [2504.10540](https://arxiv.org/abs/2504.10540) | - |
| **SpeCa** | ACM MM 2025 | DiT (FLUX/HunyuanVideo) | Predictive (Forecast-then-Verify) | **6.34× FLUX** | [2509.11628](https://arxiv.org/abs/2509.11628) | [Shenyi-Z/Cache4Diffusion](https://github.com/Shenyi-Z/Cache4Diffusion) |
| **HyCa** | 2025 | DiT (FLUX/HunyuanVideo/Qwen-Image) | Predictive (per-dim ODE mixture) | **5.55× FLUX** | [2510.04188](https://arxiv.org/abs/2510.04188) | - |
| **FEB-Cache** | 2025 | DiT | Frequency (Attn/MLP 分频) | ~2.0× | [2503.07120](https://arxiv.org/abs/2503.07120) | [aSleepyTree/EB-Cache](https://github.com/aSleepyTree/EB-Cache) |
| **FreqCa** | 2025 | DiT | Frequency (低频复用+高频预测) | ~7.14× | [2510.08669](https://arxiv.org/abs/2510.08669) | - |
| **SeaCache** | CVPR 2026 | DiT | Spectral-Evolution-Aware | ~2.5× | - | [jiwoogit/SeaCache](https://github.com/jiwoogit/SeaCache) |
| **🔥 SpectralCache** | 2026 | DiT (FLUX/PixArt) | Hybrid (TADS×CEB×FDC) | **2.46×** | [2603.05315](https://arxiv.org/abs/2603.05315) | [leeguandong/SpectralCache](https://github.com/leeguandong/SpectralCache) |
| **🔥 LayerCache** | 2026 | Flow Matching (Qwen-Image/FLUX) | Layer-Adaptive + JVP | **1.71×** | Coming soon | Coming soon |
| **MixCache** | 2025 | Video DiT | Mixture-of-Cache | ~2.2× | [2508.12691](https://arxiv.org/abs/2508.12691) | - |
| **BWCache** | 2025 | Video DiT | Block-Wise | ~2.0× | [2509.13789](https://arxiv.org/abs/2509.13789) | [hsc113/BWCache](https://github.com/hsc113/BWCache) |
| **ERTACache** | ICLR 2026 | DiT / Video DiT (Wan2.1) | Timestep-Adaptive + 残差矫正 | ~2.0× | [2508.21091](https://arxiv.org/abs/2508.21091) | [bytedance/ERTACache](https://github.com/bytedance/ERTACache) |
| **GoCache** | 2025 | DiT | Predictive (梯度补偿) | 50% blocks cached, FID -43% | [2503.05156](https://arxiv.org/abs/2503.05156) | [qiujx0520/GOC_ICCV2025](https://github.com/qiujx0520/GOC_ICCV2025) |
| **ProCache** | 2025 | DiT / PixArt-α | Constraint-aware + Selective Compute | **2.90× DiT / 1.96× PixArt** | [2512.17298](https://arxiv.org/abs/2512.17298) | [macovaseas/ProCache](https://github.com/macovaseas/ProCache) |
| **DisCa** | 2026 | Video DiT (蒸馏后) | Learnable Predictor + Distill-Compatible | **11.8×** | [2602.05449](https://arxiv.org/abs/2602.05449) | [Tencent-Hunyuan/DisCa](https://github.com/Tencent-Hunyuan/DisCa) |
| **FlowCache** | 2026 | AR Video (MAGI-1 / SkyReels-V2) | Chunkwise + KV 压缩 | 2.38× / **6.7×** | [2602.10825](https://arxiv.org/abs/2602.10825) | [mikeallen39/FlowCache](https://github.com/mikeallen39/FlowCache) |
| **AccelAes** | 2026 | DiT (Lumina-Next) | Aesthetic-Aware 时空压缩 | 2.11× (+11.9% ImageReward) | [2603.12575](https://arxiv.org/abs/2603.12575) | [xuanhuayin/AccelAes](https://github.com/xuanhuayin/AccelAes) |
| **WorldCache** | 2026 | Video World Model (Cosmos-Predict2.5) | Content-Aware + Motion-Adaptive | 2.3× | [2603.22286](https://arxiv.org/abs/2603.22286) | [umair1221/WorldCache](https://github.com/umair1221/WorldCache) |
| **HetCache** | 2026 | Video DiT (MV2V Editing) | Heterogeneous (context vs. generative tokens) | 2.67× | [2603.24260](https://arxiv.org/abs/2603.24260) | - |
| **DiffSparse** | 2026 | DiT-XL/PixArt/FLUX/Wan2.1 | Token Sparsity + Cache (learned) | 54% FLOPs↓ on PixArt | [2604.03674](https://arxiv.org/abs/2604.03674) | - |
| **Chorus** | 2026 | 4-step distilled Video DiT | **Inter-Request 三段式** | ~1.45× (+45%) | [2604.04451](https://arxiv.org/abs/2604.04451) | - |
| **X-Cache** | 2026 | AR World Model (X-world / multi-camera driving) | Cross-Chunk Block Caching + dual-metric gating | 2.6× | [2604.20289](https://arxiv.org/abs/2604.20289) | - |
| **ScalingCache** | 2026 | DiT / Wan2.1 / HunyuanVideo / FLUX | Difference Scaling + Dynamic Interval | ~2.5× video / **3.1× FLUX** | - | [Lihui-Gu/ScalingCache](https://github.com/Lihui-Gu/ScalingCache) |
| **FIS-DiT** | 2026 | Few-step Video DiT (Wan2.2 / HunyuanVideo 1.5) | Training-free Frame Interleaved Sparsity | 2.11–2.41× | [2605.11869](https://arxiv.org/abs/2605.11869) | - |
| **ECAD** | ICLR 2026 | PixArt / FLUX.1 | Static (进化搜索) | 2.58× | [2506.15682](https://arxiv.org/abs/2506.15682) | [AniAggarwal/ecad](https://github.com/AniAggarwal/ecad) |
| **SVD-Cache** | 2026 | FLUX / HunyuanVideo | Predictive (SVD 子空间) | 5.55× | [2601.07396](https://arxiv.org/abs/2601.07396) | - |
| **MeanCache** | ICLR 2026 | FLUX / Qwen-Image / HunyuanVideo | Predictive (JVP 平均速度) | **4.56×** | [2601.19961](https://arxiv.org/abs/2601.19961) | [UnicomAI/MeanCache](https://github.com/UnicomAI/MeanCache) |
| **ToPi** | 2026 | DiT (in-context) | Token 剪枝 | >30%↓ | [2602.01609](https://arxiv.org/abs/2602.01609) | - |
| **AdaCorrection** | 2026 | DiT / Video DiT | Timestep-Adaptive (偏移矫正) | ~2.0× | [2602.13357](https://arxiv.org/abs/2602.13357) | - |
| **PrediT** | 2026 | DiT / Video DiT | Predictive (线性多步) | 5.54× | [2602.18093](https://arxiv.org/abs/2602.18093) | - |
| **LESA** | 2026 | DiT / Video DiT | Predictive (KAN 多阶段) | **6.25×** | [2602.20497](https://arxiv.org/abs/2602.20497) | - |
| **SenCache** | CVPR 2026 | Wan2.1 / CogVideoX / LTX-Video | Timestep-Adaptive (敏感度) | SOTA 质量 | [2602.24208](https://arxiv.org/abs/2602.24208) | [vita-epfl/SenCache](https://github.com/vita-epfl/SenCache) |
| **RFC** | ICLR 2026 | DiT | Predictive (关系特征估计) | SOTA | - | [RFC project](https://cvlab.yonsei.ac.kr/projects/RFC/) |
| **PreciseCache** | ICLR 2026 | Wan2.1-14B / Video DiT | Video Hybrid (step+block) | ~2.6× | [2603.00976](https://arxiv.org/abs/2603.00976) | - |
| **Spectrum** | CVPR 2026 | FLUX.1 / DiT | Predictive (Chebyshev 多项式) | **4.79×** | [2603.01623](https://arxiv.org/abs/2603.01623) | [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum) |
| **TAP** | 2026 | DiT (多架构) | Token × Predictive 自适应 | — | [2603.03792](https://arxiv.org/abs/2603.03792) | - |
| **SODA** | 2026 | DiT | Hybrid (敏感度+DP 调度) | — | [2603.07057](https://arxiv.org/abs/2603.07057) | - |
| **JiT** | CVPR 2026 | FLUX.1-dev | Token / 空间 ODE 稀疏 | **7×** | [2603.10744](https://arxiv.org/abs/2603.10744) | [Wenhao-Sun77/Just-in-Time](https://github.com/Wenhao-Sun77/Just-in-Time) |
| **TimeMask** | 2026 | DiT (image+video) | Timestep (learned masking) | 1.48–2.75× | [2603.19939](https://arxiv.org/abs/2603.19939) | - |
| **SCOPE** | 2026 | MAGI-1 / SkyReels-V2 (AR video) | Video (三模调度) | 4.73× | [2604.02979](https://arxiv.org/abs/2604.02979) | - |
| **E²-CRF** | 2026 | 频域扩散模型 | Frequency (事件驱动闭环) | ~2.2× | [2604.22901](https://arxiv.org/abs/2604.22901) | - |
| **L2P-Cache** | CVPR 2026 | FLUX.1-dev / Qwen-Image | Predictive (learned linear weights) | 4.15× FLUX / 7.18× Qwen-Image | [2604.26365](https://arxiv.org/abs/2604.26365) | [Aredstone/L2P-Cache](https://github.com/Aredstone/L2P-Cache) |
| **HSA** | 2026 | Video DiT (Wan-2 / LTX-2) | Token × timestep budget + KV sync | 50% / 25% runtime Pareto | [2605.06892](https://arxiv.org/abs/2605.06892) | [project](https://ernestchu.github.io/hsa) |
| **MotionCache** | 2026 | AR 视频生成 | Video (运动感知) | — | [2605.01725](https://arxiv.org/abs/2605.01725) | - |
| **SoftCap** | 2026 | FLUX.1-dev | Timestep-Adaptive (soft-budget control) | ImageReward 0.981 @ comparable FLOPs | [2605.27075](https://arxiv.org/abs/2605.27075) | - |
| **MoECa** | 2026 | DiT-MoE | Hybrid (MoE 分支级复用) | — | [2606.15615](https://arxiv.org/abs/2606.15615) | - |
| **LearniBridge** | ICML 2026 | FLUX / HunyuanVideo / Wan2.1 | Learnable calibration (LoRA bridge) | 5.87× / 5.75× / 4.10× | [2606.26778](https://arxiv.org/abs/2606.26778) | [Iiiiiiirene/LearniBridge](https://github.com/Iiiiiiirene/LearniBridge) |
| **SyncCache** | ECCV 2026 | HunyuanVideo-Avatar / Wan-S2V | Video / modality-decoupled residual cache | 4.12× / 3.75× | [2606.30849](https://arxiv.org/abs/2606.30849) | - |
| **ACID** | 2026 | HunyuanVideo / Wan2.1 / CogVideoX | Timestep-Adaptive (critical-step 双阈值 wrapper) | 最高 2.16× vs. no-cache；+38% vs. 保守固定阈值 | [2607.12358](https://arxiv.org/abs/2607.12358) | 未开源 |
| **Kaleido** | 2026 | HunyuanVideo / Wan2.1 / CogVideoX / TurboDiffusion | Channel-wise partial-result reuse + hardware | 最高 **5.9×** vs. SOTA accelerator；16.0× energy saving（RTL 仿真） | [2607.13770](https://arxiv.org/abs/2607.13770) | 未开源 |
| **FlashDiff** | 2026 | Image / Video / Audio Diffusion | Region × Timestep + Serving | Online RCT↓ 30–97% vs. SOTA engines / throughput 1.2–2.2× | [2607.12121](https://arxiv.org/abs/2607.12121) | 未开源 |
| **CODA** | MICRO 2026 | Edge Video DiT | Compute-Cache Operator Disaggregation + CFG pipeline | 最高 1.80× / 1.74× energy efficiency vs. Vanilla-GPU（profiling + NMP 建模） | [2607.14908](https://arxiv.org/abs/2607.14908) | 未开源 |
| **DSTAR** | MICRO 2026 | 7 类 DiT（图像 / 视频 / 编辑） | Sparse attention reuse + 差分激活混合精度 + 加速器 | 7.33× / 41.89× energy vs. A100；2.54× vs. SOTA accelerator | [2607.15846](https://arxiv.org/abs/2607.15846) | 未开源 |
| **DiTango** | 2026 | 高分辨率 / 长时长 DiT（多机） | Context-Parallel × selective attention state reuse | 1.9× end-to-end / 3.2× attention（多节点近线性扩展） | [2607.15650](https://arxiv.org/abs/2607.15650) | 未开源 |
| **HeadCast** | 2026 | AR 视频 DiT（流式长视频） | Attention-head 原型分类 + head-specific KV cache 通路 | 1.62× @720P / 1.95× @1080P | [2607.20125](https://arxiv.org/abs/2607.20125) | [sjlgaga/HeadCast](https://github.com/sjlgaga/HeadCast) ![](https://img.shields.io/github/stars/sjlgaga/HeadCast.svg) |
| **EVO** | PRCV 2026 | Diffusion Policy（视觉运动控制） | 进化搜索 block × timestep 全局 cache schedule | 8.05× action generation；FLOPs 15.77G→1.96G | [2607.20293](https://arxiv.org/abs/2607.20293) | [pillom/EVO](https://github.com/pillom/EVO) ![](https://img.shields.io/github/stars/pillom/EVO.svg) |
| **CachedSearch** | 2026 | Wan / LTX / CogVideoX / Hunyuan（1.3B–14B） | Cache × test-time search（探索用 cache，胜者全算） | N=8 时以 63% 成本拿到 best-of-N 94.7% 收益；探索省 3.11× | [2607.23159](https://arxiv.org/abs/2607.23159) | - |
| **OmniCache** | 2026 | SD3 / SVD-XT / Latte | Hybrid 多维分层（Token / Frame / Block / Layered） | latency ↓ 35% / 25% / 28% | [2607.23844](https://arxiv.org/abs/2607.23844) | - |
| **FeatFix** | 2026 | 4 类图像 / 视频 backbone | Predictive 校正（verification 站点 exact-feature 复用） | 最高 6.70× vs. Vanilla | [2607.27842](https://arxiv.org/abs/2607.27842) | - |
| **OnlineCache** | 2026 | FLUX.1-dev / DiT / CogVideoX | Timestep-Adaptive（policy-gradient 学习调度 + 误差矫正） | FLUX 近 3× | [2607.29398](https://arxiv.org/abs/2607.29398) | - |
| **RACER** | 2026 | SD3.5-Large / FLUX.1-dev / Wan2.1-14B / HunyuanVideo | Predictive 闭环（双 forecast 分歧度 → 收缩 / 刷新） | 等 NFE 下全面优于最强 open-loop baseline；SD3.5 等质量更快 | [2608.01740](https://arxiv.org/abs/2608.01740) | [LiZaiyuan0619/RACER](https://github.com/LiZaiyuan0619/RACER) ![](https://img.shields.io/github/stars/LiZaiyuan0619/RACER.svg) |
| **WorldDynCache** | 2026 | HunyuanVoyager-13B / Aether-5B | Video world model 风险受控 latent dynamics 近似 | 4.92× / 2.15× | [2608.01845](https://arxiv.org/abs/2608.01845) | - |
| **EchoCache** | ACM MM 2026 | Wan2.2-S2V 等 A2V 模型 | 跨模态（音频能量引导 latent cache + 量化 cache 管理） | Wan2.2-S2V 2.46× | [2608.02474](https://arxiv.org/abs/2608.02474) | 论文声明 [IF-LAB-PKU/EchoCache](https://github.com/IF-LAB-PKU/EchoCache)（暂未公开）|

> 备注：算法类加速比对应各论文的最佳无损/近无损配置；FlashDiff 的 RCT 包含在线排队/调度收益；Kaleido 基于 16nm RTL / cycle-level 仿真，CODA 基于 RTX 4090 profiling + Ramulator / NMP RTL 建模，DSTAR 基于专用加速器实现与 A100 / SOTA accelerator 对比，均非实芯片测量；DiTango 的加速比来自多节点并行系统端到端测量。这几类数字不能与单卡算法 latency 直接横比。CachedSearch 的数字是 test-time search **预算-收益**口径，不是单次生成延迟。`未开源`状态核验于 **2026-07-19**；2026Q3 新增条目的代码仓库核验于 **2026-08-08**（RACER / EVO / HeadCast 已公开，EchoCache 论文已声明地址但仓库尚未公开）。

### 1.2 演化时间线

```
2023Q4  DeepCache                                                   (UNet feature 复用开创)
2024Q1  T-GATE                                                       (cross-attn freeze)
2024Q2  FORA / Δ-DiT                                                 (UNet cache 思路迁移到 DiT)
2024Q3  PAB                                                          (视频 DiT 金字塔广播)
2024Q4  TeaCache / ToCa / HarmoniCa                                  (timestep 阈值 / token 级 / learning)
2025Q1  TaylorSeer / BlockDance / GoCache                            (Cache-then-Forecast 开创 / STSS block / 梯度补偿)
2025Q2  MagCache / Chipmunk / LazyDiT / ProfilingDiT / AB-Cache     (幅值律 / 稀疏增量 / learned skip / FG-BG profile / Adams-Bashforth)
2025Q3  HiCache / FoCa / ClusCa / SpeCa / ERTACache                 (Hermite / Feature-ODE / token cluster / speculate-verify / 残差矫正)
2025Q4  FreqCa / FEB-Cache / DiCache / HyCa / ProCache / ECAD       (频域 / 自触发 / per-dim ODE mixture / constraint-aware / 进化调度搜索)
2026Q1  SeaCache / SpectralCache / LayerCache / DisCa / FlowCache    (频谱演化 / 频域 hybrid / 层异质 + JVP / distill 兼容 / AR video chunkwise)
2026Q1  SVD-Cache / MeanCache / RFC / SenCache / Spectrum / JiT      (SVD 子空间预测 / JVP 平均速度 / 关系特征估计 / 敏感度驱动 / Chebyshev / 空间 ODE)
2026Q1  PrediT / LESA / TAP / SODA / TimeMask / PreciseCache        (线性多步 / KAN 多阶段 / token 自适应预测 / 敏感度 DP / learned masking / 视频精准缓存)
2026Q1  AdaCorrection / ToPi                                         (偏移自适应矫正 / in-context token 剪枝)
2026Q2  AccelAes / WorldCache / HetCache / DiffSparse / Chorus       (aesthetic-aware / video world model / V2V 异质 / token sparsity / inter-request)
2026Q2  X-Cache / ScalingCache / FIS-DiT / SCOPE / E²-CRF           (跨 chunk block cache / 差分尺度 + 动态间隔 / 帧交错稀疏 / 三模 AR 调度 / 频域事件驱动)
2026Q2  L2P-Cache / HSA / MotionCache / SoftCap                      (可学习线性预测 / 异构步预算 / 运动感知 AR cache / 软预算控制)
2026Q2  MoECa / LearniBridge / SyncCache                             (MoE 分支级复用 / LoRA 轻量校准 / 音频驱动人像模态解耦 cache)
2026Q3  ACID / Kaleido                                               (临界步双阈值 / 通道级局部结果复用 + 专用加速器)
2026Q3  FlashDiff / CODA / DSTAR / DiTango                           (语义区域复用与服务调度 / compute-cache 解耦 + 近存计算 / 时空冗余 + 加速器 / 并行 attention state 复用)
2026Q3  OmniCache / FeatFix / RACER / OnlineCache                    (多维分层复用 / verify-then-correct / 分歧度闭环控制 / 学习式在线调度)
2026Q3  EchoCache / WorldDynCache / HeadCast / EVO / CachedSearch    (音频能量跨模态 / world model 风险受控 / AR head 级 KV 通路 / diffusion policy 进化调度 / cache × test-time search)
```

## 2. 按缓存 / 复用粒度分类（What is cached or reused）

本节从 **"到底在缓存或复用什么对象"** 的角度做一次正交切分，和 §3 的 "调度策略" 配合使用。每个方法通常在一个主要粒度上做文章，少数混合方法会跨多个粒度（见 §2.9 矩阵）。

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
| **SpeCa** | 未来 step 的 feature 预测值 | 投机预测 + parameter-free 验证器接受/拒绝 |
| **ERTACache** | 整步 residual 输出 | 可复用步离线 profile + 时步调整 + 闭式残差矫正 |
| **ProCache** | 整步 feature | constraint-aware 非均匀调度 + 深层 block / 高重要性 token 选择性重算 |
| **DisCa** | 整步 feature（蒸馏后模型） | 轻量可学习 predictor + Restricted MeanFlow 蒸馏兼容 |
| **Chorus** | 整步 latent feature | **跨请求**复用：早期 full reuse + 中段 region-specific + Token-Guided Attention Amplification |
| **ScalingCache** | 整步 activation | 离线 profile 冗余 + Dynamic Interval + Difference Scaling 复用 |
| **SoftCap** | 整步 full/cached 决策状态 | Trajectory Drift Observer + soft-budget PI controller 动态调 full-trigger 阈值 |
| **SVD-Cache** | 整步 feature (SVD 分解) | 主子空间 EMA 预测 + 残差子空间直接复用 |
| **MeanCache** | 整步 velocity (JVP) | 平均速度外推 + Peak-Suppressed 调度 |
| **SenCache** | 整步 feature | 基于输出敏感度 per-sample 动态缓存 |
| **AdaCorrection** | 整步 cached activation | 检测 offset drift + 逐层逐步插值矫正 |
| **RFC** | 整步 feature | 关系特征估计 + 关系调度触发 |
| **ECAD** | 整步 feature | 进化搜索 Pareto-optimal cache schedule |
| **ACID** | 基础方法已有的整步 residual / feature | 监测 drift signal 变化率，在 critical step 用低阈值、稳定步用高阈值 |
| **OnlineCache** | 整步 feature | policy-gradient 学到的动态调度策略 + 误差矫正器双层联合优化 |
| **RACER** | 整步 feature 的两路 forecast | 用两路预测的分歧度作可靠性信号：不确定处向最近实算特征收缩，最危险步刷新并延后偿还 |
| **WorldDynCache** | world model 的 latent 转移状态 | 风险估计器 + condition/phase-aware lifted latent surrogate 近似演化 |
| **EchoCache** | 整步 latent（A2V） | 音频时频能量作 saliency anchor 引导 latent 更新 + 量化 cache 管理 |
| **CachedSearch** | 已有 cache 方法的整步状态 | test-time search 中所有候选激进缓存探索，仅胜者全算重生成 |

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
| **BlockDance** | 结构焦点块在去噪后期的 STSS 特征 |
| **ProfilingDiT** | 前景/背景倾向的 block 分组,背景块可 cache |
| **GoCache** | block 输出 + 缓存梯度队列 (50% blocks 缓存) |
| **DiffSparse** | 层 × token 联合稀疏（DP solver 学习每层激活率）|
| **X-Cache** | AR 视频 chunk 间 block residual（dual-metric gating，KV update chunk 必算）|
| **TimeMask** | 每 timestep 学习 block-level mask，决定执行/跳过 |
| **SODA** | 敏感度建模 + DP 跨层最优 cache 间隔 + 统一 pruning |
| **PreciseCache** | LFCache (step-wise) + BlockCache (block-wise) 双层 |
| **SyncCache** | 音频驱动人像 DiT 的 heavy DiT block residual；复用稳定 inter-block residual，轻量 audio block 持续重算 |
| **FeatFix** | 固定稀疏 layer–timestep 站点上的**完整 block 精确输出**，用来整块替换 draft 输出（不做 token / channel 部分替换）|
| **EVO** | diffusion policy 的 block × timestep 格点 cache 状态（进化搜索出的全局 schedule）|
| **OmniCache** (Block / Layered) | block 输出 + 跨步的 model-layer 级冗余（Layered Cache）|

### 2.3 Attention Cache（注意力模块）

**缓存对象**：attention 的输出、attention map、或 KV。基于 "attention 冗余度比 MLP 更高" 的观察。

| 方法 | 缓存什么 |
|------|---------|
| **T-GATE** | cross-attention 输出（收敛点后 freeze）|
| **PAB** | self/cross/temporal attention 各自按类型广播 |
| **FEB-Cache** (Attn 分支) | 后期阶段的 attention 输出（低频结构）|
| **FasterCache** (attention 部分) | attention feature 跨步复用 |
| **CODA** (cache path) | 复用跨时步 attention 输出；对应的读取 / 缩放 / 融合等 cache operator 合并后由 DIMM-NMP 执行 |
| **DSTAR** | 稀疏 attention 复用：只算变化显著的 attention 部分，其余复用旧结果（配差分激活混合精度量化）|
| **DiTango** | Context-Parallel 下各 sequence partition 的 attention state；低贡献远端 partition 复用历史结果，高贡献近邻 partition 实算 |
| **HeadCast** | AR 视频 DiT 的 **per-head KV cache**：按 Sink / Dummy / Spatial / Global 四类原型分通路管理，Global head 完整保留 |

### 2.4 MLP / FFN Cache

**缓存对象**：transformer 中 MLP / FFN 模块的输出。

| 方法 | 缓存什么 |
|------|---------|
| **FEB-Cache** (MLP 分支) | 早期阶段的 MLP 输出（高频细节）|
| **FORA** (MLP 部分) | MLP 在固定区间内复用 |
| **CODA** (cache path) | 复用跨时步 FFN 输出；对应的 memory-bound cache operator 与 xPU dense compute 解耦并流水重叠 |

### 2.5 Fine-Grained Cache（token / region / channel）

**缓存对象**：不再把整步或整层视为同质单元，而是在 token、语义 region 或 channel / partial result 粒度选择性复用。

| 方法 | 缓存什么 |
|------|---------|
| **ToCa** | 低敏感度 token 的每层激活 |
| **DuCa** | token × layer 双层 |
| **FastCache** | 静态 token 用学习的线性近似映射 |
| **Chipmunk** | 低贡献 activation 的 column-sparse cache |
| **ClusCa** | KMeans 对 token 聚类,每簇仅算 1 个 token,簇内传播 |
| **HetCache** | 视频编辑 token 拆 context / generative 两类,只缓存 context |
| **AccelAes** | 按 cross-attn aesthetic signal 选 token，低相关区域压缩 |
| **DiffSparse** | learned per-layer token sparsity allocation |
| **FIS-DiT** | few-step 视频在帧位置上的 frame slice 稀疏（稳定 block 算子集，敏感 block 全算）|
| **HSA** | spatiotemporal token 的异构 denoising step 预算 + KV-cache 同步 |
| **JiT** | 空间稀疏 anchor token 子集 → micro-flow ODE 传播 |
| **ToPi** | in-context reference token 敏感度剪枝 |
| **TAP** | 每 token 自选最优预测器 (proxy loss 最小化) |
| **MoECa** | DiT-MoE 的 expert branch 级 token 跨步复用 |
| **FlashDiff** | 语义 latent patch 在稳定后跳步，直接复用相邻 timestep 的 prior state |
| **Kaleido** | 相似邻接 token 的 channel 级 partial attention / GEMM 结果复用 |
| **OmniCache** (Token / Frame) | intra-frame 相似 token + inter-frame / motion 冗余帧特征，用相似度匹配挑可缓存项并按原位恢复 |

### 2.6 Frequency-Band Cache（频带分解）

**缓存对象**：对特征做频域分解后的低频 / 高频分量，分别缓存。

| 方法 | 缓存什么 |
|------|---------|
| **FEB-Cache** | Attn 偏低频 / MLP 偏高频，分阶段切换对象 |
| **FreqCa** | 低频 reuse + 高频用二阶 Hermite 外推，CRF 残差降内存 99% |
| **SeaCache** | 跟踪频谱演化触发刷新 |
| **SpectralCache** (本作) | 低频 γ=0.8 严 / 高频 γ=1.5 松 的非对称阈值 |
| **E²-CRF** | 频谱局部化 + 镜像对称降维 + 事件驱动 KV cache |
| **Spectrum** | Chebyshev 多项式全局频域近似（ridge regression 拟合）|

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
| **HyCa** | 按 feature 维度拆 ODE,每维自选解算器 |
| **ERTACache** | 显式拆分 "feature shift + step amplification" 两类误差,解析闭式矫正 |
| **GoCache** | 缓存与重算之间的特征差作 "梯度",inflection-aware 加权回传补偿 |
| **SVD-Cache** | SVD 残差子空间（低能量直接复用）|
| **MeanCache** | JVP 速度残差（平均速度构建）|
| **PrediT** | 线性多步残差外推 + corrector |
| **L2P-Cache** | 历史特征轨迹的可学习线性组合残差 |
| **LearniBridge** | 用低秩 LoRA bridge 校准跨 timestep cache 误差 |
| **SyncCache** | 模态解耦的 inter-block residual 复用 |
| **FeatFix** | 在 verification 站点把 draft 残差**归零重置**（用同一入态的精确输出替换），抑制下游误差 |
| **RACER** | 对不确定的 forecast 残差做**收缩**（向最近实算特征插值），有确定性误差界 |
| **WorldDynCache** | latent transition 的近似缺陷（approximation defect）作风险量，用 exact anchor 反事实校准 |

### 2.9 缓存 / 复用粒度 × 调度策略 交叉矩阵

列 = §3 的调度策略；行 = §2 的缓存 / 复用粒度。◆ = 主要归属，○ = 次要命中。

| 粒度 \ 策略 | Static | Timestep-Adaptive | Layer-Adaptive | Predictive | Fine-Grained | Frequency-Aware | CFG | Hybrid |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Step Cache**       | DeepCache / FORA / ECAD ◆ | TeaCache / FBCache / MagCache / EasyCache / ERTACache / ProCache / ScalingCache / SenCache / AdaCorrection / SoftCap / ACID / OnlineCache ◆ | — | TaylorSeer / HiCache / AB-Cache / FoCa / SpeCa / DisCa / SVD-Cache / MeanCache / Spectrum / PrediT / LESA / RFC / L2P-Cache / LearniBridge / FeatFix / RACER ◆ | — | — | — | FasterCache ○ / Chorus (inter-req) ○ / CachedSearch (test-time search) ○ / EchoCache (cross-modal) ○ / WorldDynCache ○ |
| **Block Cache**      | Δ-DiT ◆ | Cache Me if You Can / BlockDance / TimeMask ◆ | DBCache / Skip-DiT / HarmoniCa / ProfilingDiT / GoCache / DiffSparse / SODA / EVO / **LayerCache** ◆ | FeatFix (block-level 精确校正) ○ | — | — | — | BWCache ○ / X-Cache (AR-chunk) ○ / PreciseCache ○ / SyncCache ○ / OmniCache ○ |
| **Attention Cache**  | T-GATE ◆ | — | — | — | DSTAR (sparse attn reuse) ○ / DiTango (partition) ○ / HeadCast (per-head KV) ○ | FEB-Cache (Attn) ○ | — | PAB ◆ / FasterCache ○ / CODA (system) ○ / DSTAR (system) ◆ / DiTango (parallel system) ◆ |
| **MLP Cache**        | FORA (MLP) ◆ | — | — | — | — | FEB-Cache (MLP) ◆ | — | CODA (system) ○ |
| **Fine-Grained**     | — | Chipmunk ◆ | — | — | ToCa / DuCa / FastCache / ClusCa / HetCache / AccelAes / DiffSparse / FIS-DiT / HSA / JiT / ToPi / TAP / FlashDiff (region) / Kaleido (channel, within-step reuse) / OmniCache (token+frame) ◆ | — | — | MoECa (MoE branch) ○ |
| **Frequency Band**   | — | — | — | FreqCa (高频预测) ○ / Spectrum ○ | — | FreqCa / SeaCache / FEB-Cache / E²-CRF / **SpectralCache** ◆ | — | **SpectralCache** ○ |
| **CFG Branch**       | — | — | — | — | — | FasterCache (CFG+freq) ○ | CFG-Cache ◆ | — |
| **Residual**         | Δ-DiT ◆ | Chipmunk / ERTACache ○ | **LayerCache** (JVP) ◆ | AB-Cache / FoCa / HiCache / HyCa / GoCache / SVD-Cache / MeanCache / PrediT / L2P-Cache / LearniBridge / FeatFix / RACER ◆ | — | — | — | SyncCache ○ / WorldDynCache ○ |

> **怎么读这张表**：
> - 横向看：一个调度策略下都有哪些缓存 / 复用粒度的代表。
> - 纵向看：同一粒度下不同调度思路的演化。
> - **LayerCache** 同时命中 *Block / Residual* 粒度 + *Layer-Adaptive / Predictive* 策略（所以在 Hybrid 意义上是"层粒度 + 预测"）。
> - **SpectralCache** 同时命中 *Frequency Band* 粒度 + *Frequency-Aware / Hybrid* 策略。
> - **Kaleido** 是同一 timestep 内的 partial-result reuse（non-CTC），与传统 cross-timestep feature cache 互补。**DSTAR** 的 sparse attention reuse、**DiTango** 的 partition state reuse、**HeadCast** 的 per-head KV 通路同属"非纯 cross-timestep"的复用轴。
> - **FeatFix / RACER** 都不新造预测器，而是改"预测结果怎么用"：FeatFix 在稀疏站点用精确输出整块重置 draft 残差，RACER 用双 forecast 的分歧度决定信任多少并在危险步刷新 —— 可视为预测类方法的**误差控制层**。
> - **CachedSearch** 是唯一把 cache 用在 **test-time search 预算分配**上的工作：不追求单次生成更快，而是让"广探索 + 胜者全算"的总收益更高，与上面 8 列正交。
> - **Service-Level 维度**独立于上面 8 列：**Chorus** 做跨请求 feature 复用；**FlashDiff** 做请求内 region state 复用并把节省出的算力重排给并发请求；**DiTango** 把复用决策与多机通信拓扑绑定，详见 §3.10。

## 3. 按调度策略详述（How to decide）

### 3.1 Static Caching（固定调度）

固定复用区间 / 固定层集合，**无运行时决策**。

* **DeepCache**：
  * 地址：https://github.com/horseee/DeepCache ![](https://img.shields.io/github/stars/horseee/DeepCache.svg)
  * 论文：[CVPR 2024](https://arxiv.org/abs/2312.00858)
  * 简介：首个系统化利用扩散模型时序冗余的 training-free 方法。基于 UNet skip connection 观察：高层特征跨相邻步变化平缓。DeepCache 跨步复用 UNet 上采样路径的 deep feature，浅层每步重算。在 SD 1.5/2.1 上可取得约 2.3× 加速，几乎无损。**仅适用于 UNet，DiT 不适用**。

* **FasterDiffusion**：
  * 地址：https://github.com/hutaiHang/Faster-Diffusion ![](https://img.shields.io/github/stars/hutaiHang/Faster-Diffusion.svg)
  * 论文：[NeurIPS 2024 / arXiv 2312.09608](https://arxiv.org/abs/2312.09608)
  * 简介：发现 UNet encoder 对相邻 step 的输出非常相似，提出 encoder propagation：跨步复用 encoder 输出，仅 decoder 继续更新。同时引入 parallel decoding 降低串行开销。DiT 变体参见 [sen-mao/FasterDiffusion-DiT](https://github.com/sen-mao/FasterDiffusion-DiT)。

* **T-GATE**：
  * 地址：https://github.com/HaozheLiu-ST/T-GATE ![](https://img.shields.io/github/stars/HaozheLiu-ST/T-GATE.svg)
  * 论文：[arXiv 2404.02747](https://arxiv.org/abs/2404.02747)
  * 简介：发现 cross-attention 在早期去噪阶段即收敛，之后几乎不变。T-GATE 在收敛点直接 freeze cross-attention 输出并跨步复用。适用 SD、PixArt、LCM。V2 进一步支持 DiT。

* **FORA** (First-Order Residual Approximation)：
  * 地址：https://github.com/prathebaselva/FORA ![](https://img.shields.io/github/stars/prathebaselva/FORA.svg)
  * 论文：[arXiv 2407.01425](https://arxiv.org/abs/2407.01425)
  * 简介：把 DeepCache 思路迁移到 DiT，固定间隔复用 self-attn / MLP 输出。是 DiT cache 领域最早的 baseline 之一。

* **Δ-DiT**：
  * 论文：[arXiv 2406.01125](https://arxiv.org/abs/2406.01125)
  * 简介：缓存 residual 增量而非绝对值，并根据生成阶段（布局 / 细节）动态调整不同 block 的缓存侧重。

* **PAB (Pyramid Attention Broadcast)**：
  * 地址：https://github.com/NUS-HPC-AI-Lab/VideoSys ![](https://img.shields.io/github/stars/NUS-HPC-AI-Lab/VideoSys.svg)
  * 论文：[ICLR 2025 / arXiv 2408.12588](https://arxiv.org/abs/2408.12588)
  * 简介：视频 DiT 的 attention 差分呈 U 形，空间 / 时间 / cross attention 稳定性不同。PAB 按注意力类型设置不同广播半径（金字塔式），在 Open-Sora / Latte / Open-Sora-Plan 上达到 21.6 FPS 实时生成，10.6× 加速。

* **ECAD** (Evolutionary Caching, ICLR 2026)：
  * 地址：https://github.com/AniAggarwal/ecad ![](https://img.shields.io/github/stars/AniAggarwal/ecad.svg)
  * 论文：[ICLR 2026 / arXiv 2506.15682](https://arxiv.org/abs/2506.15682)
  * 简介：用**遗传算法**在少量 calibration prompt 上自动搜索每个模型的最优 cache schedule，生成 Pareto 前沿（质量 vs. 延迟）。无需修改网络参数，可泛化到未见分辨率和模型变体。PixArt-α 上 2.58× 加速，相对前 SOTA 提升 4.47 FID。

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
  * 地址：https://github.com/H-EmbodVis/EasyCache ![](https://img.shields.io/github/stars/H-EmbodVis/EasyCache.svg)
  * 论文：[ICCV 2025 / arXiv 2507.02860](https://arxiv.org/abs/2507.02860)
  * 简介：runtime adaptive self-correct：相对变换率 + 累计偏差双指标，自适应调整阈值，2.1–3.3× 加速。

* **LazyDiT**：
  * 地址：https://github.com/shawnricecake/lazydit ![](https://img.shields.io/github/stars/shawnricecake/lazydit.svg)
  * 论文：[AAAI 2025 / arXiv 2412.12444](https://arxiv.org/abs/2412.12444)
  * 简介：在每个 transformer layer 前插入线性预测器，用一阶 Taylor 近似 learn 相似度，决定是否跳过该层计算。

* **Chipmunk**：
  * 地址：https://github.com/sandyresearch/chipmunk ![](https://img.shields.io/github/stars/sandyresearch/chipmunk.svg)
  * 论文：[arXiv 2506.03275](https://arxiv.org/abs/2506.03275)
  * 简介：发现 5–25% 的 activation 占 70–90% 变化量，提出 column-sparse activation cache，硬件友好。

* **Cache Me if You Can (Block Cache)**：
  * 论文：[CVPR 2024 / arXiv 2312.03209](https://arxiv.org/abs/2312.03209)
  * 简介：每个 block 有独立阈值，相对变化超阈值才刷新，早期的 block-wise 阈值工作。

* **ERTACache** (ICLR 2026)：
  * 地址：https://github.com/bytedance/ERTACache ![](https://img.shields.io/github/stars/bytedance/ERTACache.svg)
  * 论文：[arXiv 2508.21091](https://arxiv.org/abs/2508.21091)
  * 简介：字节跳动提出。显式把缓存误差拆成 **feature shift error**（特征漂移）与 **step amplification error**（步放大）两部分，用三件套矫正：离线 residual profiling 挑可复用步 + 轨迹感知时步调整 + 闭式残差修正。在 Wan2.1 上 2.0× 加速，VBench 几乎无损。

* **ProCache** (2025-12)：
  * 地址：https://github.com/macovaseas/ProCache ![](https://img.shields.io/github/stars/macovaseas/ProCache.svg)
  * 论文：[arXiv 2512.17298](https://arxiv.org/abs/2512.17298)
  * 简介：把"何时刷新"建模为 constraint-aware 调度搜索问题，生成与 DiT 时序特征对齐的**非均匀**激活节奏，避免固定间隔 cache 与 DiT 动力学失配；并在深层 block / 高重要性 token 上做选择性重算抑制误差累积。**training-free**，PixArt-α 1.96× / DiT 2.90×。

* **ScalingCache** (2026)：
  * 地址：https://github.com/Lihui-Gu/ScalingCache
  * 论文：OpenReview (uXmbrTlko7)
  * 简介：用**离线轻量分析**少量样本拿到激活冗余画像，再在推理时**动态决定缓存间隔**（Dynamic Interval Caching）+ Difference Scaling 复用旧激活。Wan2.1 / HunyuanVideo 上约 **2.5× 加速 + ≤0.5% VBench 降幅**，FLUX 上 **3.1× 近无损**，LPIPS 相对前 SOTA 降 45%。

* **SenCache** (CVPR 2026)：
  * 地址：https://github.com/vita-epfl/SenCache ![](https://img.shields.io/github/stars/vita-epfl/SenCache.svg)
  * 论文：[CVPR 2026 / arXiv 2602.24208](https://arxiv.org/abs/2602.24208)
  * 简介：从理论出发，用模型输出对噪声 latent 和 timestep 扰动的**敏感度**来形式化缓存误差。据此构建 per-sample 动态缓存策略——敏感度低（特征稳定）的步自动缓存，敏感度高的步刷新。Wan2.1 / CogVideoX / LTX-Video 上在同等计算预算下达到 SOTA 视觉质量。

* **AdaCorrection** (2026-02)：
  * 论文：[arXiv 2602.13357](https://arxiv.org/abs/2602.13357)
  * 简介：推理时轻量框架，动态检测缓存激活的 **offset drift** 并逐层逐步自适应矫正。Offset Estimation Module (OEM) 通过时空偏差统计量化偏移量，Adaptive Correction Module (ACM) 在缓存值与新鲜计算之间插值。

* **TimeMask** (2026-03)：
  * 论文：[arXiv 2603.19939](https://arxiv.org/abs/2603.19939)
  * 简介：对预训练扩散模型**每个 timestep 独立学习** block 级 mask，决定哪些 block 执行、哪些通过特征复用跳过。相比全局优化方法，按 timestep 独立优化 mask 更省显存。1.48–2.75× 加速。

* **SoftCap** (2026-05)：
  * 论文：[arXiv 2605.27075](https://arxiv.org/abs/2605.27075)
  * 简介：给 cache-based DiT 推理加一层 **soft-budget control**。Trajectory Drift Observer 用轻量 hidden-state 统计估计局部 cache 风险，Soft-Budget PI Controller 根据实际 compute 与参考 profile 的偏差动态调整 full-step 触发阈值。FLUX.1-dev 上在接近相同 FLOPs 下优于 SpeCa，ImageReward 0.981、LPIPS-Full 0.498。

* **ACID** (Adaptive Caching for vIDeo Generation, 2026-07)：
  * 论文：[arXiv 2607.12358](https://arxiv.org/abs/2607.12358)
  * 简介：指出 TeaCache / EasyCache / DiCache 的质量-速度折中很大程度来自**全轨迹固定阈值**。ACID 不替换原方法的 drift signal，而是监测信号的局部变化率：critical step 切到低阈值保质量，稳定区间切到高阈值激进缓存。它是 training-free、signal-agnostic 的 wrapper，无需改 backbone；在 HunyuanVideo + TeaCache 上相对无 cache 达 **2.16×**，比保守固定阈值再快 **38%**，同时 PSNR / SSIM / LPIPS 变化很小。

* **OnlineCache** (2026-07)：
  * 论文：[arXiv 2607.29398](https://arxiv.org/abs/2607.29398)
  * 简介：不再手调阈值，而是用 **policy gradient 学习一个动态 timestep 级缓存策略**，并配一个误差矫正器补偿缓存引入的偏差。两者在 **bilevel 优化**框架下联合训练：policy 以全局生成质量为目标，corrector 以局部误差最小化为目标，从而在样本与时步两个维度上自动分配算力。FLUX.1-dev 上近 **3×** 加速且保真度基本不掉，DiT / CogVideoX 上同样稳定优于既有 cache baseline。

### 3.3 Layer-Adaptive（深度自适应）

在**层深度维度**决定哪些层算 / 哪些层缓存，代表了"不同层对 cache 敏感度不同"的洞察。

* **HarmoniCa**：
  * 地址：https://github.com/ModelTC/HarmoniCa ![](https://img.shields.io/github/stars/ModelTC/HarmoniCa.svg)
  * 论文：[arXiv 2410.01723](https://arxiv.org/abs/2410.01723)
  * 简介：首个 learning-based cache schedule 工作。在完整 denoising trajectory 上训练 cache controller，解决 train-inference mismatch。

* **AdaCache**：
  * 地址：https://adacache-dit.github.io/
  * 论文：[arXiv 2411.02397](https://arxiv.org/abs/2411.02397)
  * 简介：视频 DiT 的 content-adaptive schedule——**每个 video 都有独立的 cache 计划**。结合 residual 变化 + motion regularization，在 Open-Sora 上达到 4.49× 加速。

* **DBCache (Dual Block Cache)**：
  * 地址：https://github.com/vipshop/cache-dit ![](https://img.shields.io/github/stars/vipshop/cache-dit.svg)（唯品会 cache-dit 工程库的核心调度组件，无独立论文）
  * 简介：DiT block stack 分三段：**Probe（前段全算）→ Main（中段阈值缓存）→ Corrector（尾段纠正）**。典型的概率-决策-纠错架构。

* **Skip-DiT**：
  * 地址：https://github.com/OpenSparseLLMs/Skip-DiT ![](https://img.shields.io/github/stars/OpenSparseLLMs/Skip-DiT.svg)
  * 论文：[ICCV 2025 / arXiv 2411.17616](https://arxiv.org/abs/2411.17616)
  * 简介：借鉴 long-skip-connection 思想，深层做 static cache，浅层每步 update，解决深层 DiT 稳定性问题。

* **BlockDance (CVPR 2025)**：
  * 论文：[arXiv 2503.15927](https://arxiv.org/abs/2503.15927)
  * 简介：发现去噪后期**结构焦点 block** 的 spatio-temporal 特征高度相似（**STSS** = Structurally Similar Spatio-Temporal），只对这些 block 做选择性 cache 而非一刀切 block-wise 缓存。BlockDance-Ada 变体引入轻量决策网络实现 instance-level 自适应。在 DiT-XL/2 (37.4%) / PixArt-α (25.4%) / Open-Sora (34.8%) 上 25–50% 加速。

* **ProfilingDiT (ICCV 2025)**：
  * 地址：https://github.com/GeekGuru123/ProfilingDiT ![](https://img.shields.io/github/stars/GeekGuru123/ProfilingDiT.svg)
  * 论文：[arXiv 2504.03140](https://arxiv.org/abs/2504.03140)
  * 简介：发现 DiT 大部分层对**前景/背景**有稳定偏好，且噪声相似度随去噪递增。据此 profile 把 block 拆成 FG-focused 与 BG-focused 两组，**前景块每步重算，背景块激进缓存**。Wan2.1 上 2.01× 加速。

* **GoCache** (Gradient-Optimized Cache, 2025-03)：
  * 地址：https://github.com/qiujx0520/GOC_ICCV2025 ![](https://img.shields.io/github/stars/qiujx0520/GOC_ICCV2025.svg)
  * 论文：[arXiv 2503.05156](https://arxiv.org/abs/2503.05156)
  * 简介：针对 cached block 引入的近似误差，构建**缓存-重算特征差**作为梯度信号，配合 **inflection-aware** 调度（在去噪轨迹拐点处加权回传）补偿误差。50% blocks 缓存下 IS +26.3% / FID -43%，开销持平。

* **DiffSparse** (2026-04)：
  * 论文：[arXiv 2604.03674](https://arxiv.org/abs/2604.03674)
  * 简介：把"每层每 token 是否激活"建模为可微优化问题——learnable controller + DP solver 联合求解层间 token sparsity 分配，端到端学习最优 cache + 重算策略。在 DiT-XL/2 / PixArt-α / FLUX / Wan2.1 通用，PixArt-α 上 20 步 54% FLOPs 削减且质量提升。

* **SODA** (2026-03)：
  * 论文：[arXiv 2603.07057](https://arxiv.org/abs/2603.07057)
  * 简介：离线做细粒度**敏感度建模**（跨 timestep × layer × module），定义误差度量并用**动态规划**求解最优 cache 间隔分配。同时把 token pruning 的时机和比率也纳入敏感度指导，统一了 caching + pruning 的调度框架。

* **🔥 LayerCache (CVPR 2026)**：
  * 地址：Coming soon
  * 简介：发现 flow matching 模型中 transformer 的**层组速度异质性**——Shallow / Middle / Deep 有不同的稳定度：浅层稳定可激进缓存（98%），中层中等（52%），深层高度易变（0% 缓存）。提出 **3D schedule (timestep × layer group × JVP span K)** + greedy budget allocation + JVP-based forecasting。在 Qwen-Image 上 1.71× 加速，PSNR 34.16，显著优于 MeanCache baseline。

### 3.4 Predictive / Cache-then-Forecast（预测类）

把 cache 升级为**数值积分 / 多项式外推**：用历史 step 的特征预测未来 step。

* **TaylorSeer**：
  * 地址：https://github.com/Shenyi-Z/TaylorSeer ![](https://img.shields.io/github/stars/Shenyi-Z/TaylorSeer.svg)
  * 论文：[ICCV 2025 / arXiv 2503.06923](https://arxiv.org/abs/2503.06923)
  * 简介：**Cache-then-Forecast 范式开创**。用多步历史特征做差分近似各阶导数，Taylor 级数外推未来 step 的特征。奠定了后续预测类方法的理论基础。

* **HiCache**：
  * 地址：https://github.com/fenglang918/HiCache ![](https://img.shields.io/github/stars/fenglang918/HiCache.svg)
  * 论文：[arXiv 2508.16984](https://arxiv.org/abs/2508.16984)
  * 简介：发现 DiT 特征导数的近似呈多元高斯特征，改用 **Hermite 多项式**（高斯共轭的理论最优基）替换 Taylor 基，plug-and-play，显著提升稳定性。

* **FoCa** (Forecast then Calibrate)：
  * 论文：[arXiv 2508.16211](https://arxiv.org/abs/2508.16211)
  * 简介：把 hidden feature 序列显式建模为 **feature-ODE**，用预测-校正求解器直接当 cache 外推器。大步长下依然稳定，FLUX 5.50×、HunyuanVideo 6.45× 加速。

* **AB-Cache**：
  * 论文：[arXiv 2504.10540](https://arxiv.org/abs/2504.10540)
  * 简介：**Adams-Bashforth** 多步法，解释了 U 形相似度现象的数学根源——相邻 step 输出之间的线性关系，误差界 O(h^k)，在 FLUX.1-dev / HunyuanVideo 上约 3× 加速。

* **SpeCa (ACM MM 2025)**：
  * 地址：https://github.com/Shenyi-Z/Cache4Diffusion ![](https://img.shields.io/github/stars/Shenyi-Z/Cache4Diffusion.svg)（与 ClusCa / TaylorSeer 共用的统一仓库）
  * 论文：[arXiv 2509.11628](https://arxiv.org/abs/2509.11628)
  * 简介：把 LLM **Speculative Decoding** 搬到 Diffusion Feature Cache——先用参考 timestep 预测 upcoming feature（forecast），再用 **parameter-free verifier** 接受/拒绝（verify），配合 sample-adaptive 计算预算。FLUX 上 **6.34× 加速**（质量降 5.5%），DiT 上 7.3×，HunyuanVideo 上 6.1×。相对 TaylorSeer 等"无验证"预测类方法补上了精度保证环节。

* **HyCa (Hybrid ODE Cache)**：
  * 论文：[arXiv 2510.04188](https://arxiv.org/abs/2510.04188)
  * 简介：发现不同 feature 维度演化行为差异极大，**对所有维度套同一个 ODE solver 不是最优**。把 hidden feature 建模为多维 ODE 混合，**每个维度自选最合适的数值解算器**。training-free，FLUX 5.55× / HunyuanVideo 5.56× / Qwen-Image(Edit) 6.24× 加速。

* **DisCa** (Distillation-Compatible Learnable Feature Caching, 2026-02)：
  * 地址：https://github.com/Tencent-Hunyuan/DisCa ![](https://img.shields.io/github/stars/Tencent-Hunyuan/DisCa.svg)
  * 论文：[arXiv 2602.05449](https://arxiv.org/abs/2602.05449)
  * 简介：突破 training-free cache 在**蒸馏后视频 DiT** 上掉点严重的瓶颈。一是用**轻量可学习神经预测器**取代手工启发，更准地外推高维特征；二是配套 **Restricted MeanFlow** 蒸馏策略，使 step distillation × feature cache 共存近无损。报告 **11.8×** 加速，是当前蒸馏 + cache 联合栈最强的单点。

* **SVD-Cache** (2026-01)：
  * 论文：[arXiv 2601.07396](https://arxiv.org/abs/2601.07396)
  * 简介：对 DiT 特征做 **SVD 分解**，拆成主子空间（平滑可预测）和残差子空间（易变但低能量）。主子空间用 EMA 预测，残差子空间直接复用。**近无损 5.55×** 加速，兼容蒸馏 / 量化 / 稀疏注意力。

* **MeanCache** (ICLR 2026)：
  * 地址：https://github.com/UnicomAI/MeanCache ![](https://img.shields.io/github/stars/UnicomAI/MeanCache.svg)
  * 论文：[ICLR 2026 / arXiv 2601.19961](https://arxiv.org/abs/2601.19961)
  * 简介：引入**平均速度**视角——用缓存的 Jacobian-Vector Product (JVP) 从瞬时速度构建区间平均速度，缓解局部误差累积。配合 **Peak-Suppressed Shortest Path** 轨迹稳定性调度做预算约束下的最优复用。FLUX 4.12× / Qwen-Image 4.56× / HunyuanVideo 3.59×。

* **Spectrum** (CVPR 2026)：
  * 地址：https://github.com/hanjq17/Spectrum ![](https://img.shields.io/github/stars/hanjq17/Spectrum.svg)
  * 论文：[CVPR 2026 / arXiv 2603.01623](https://arxiv.org/abs/2603.01623)
  * 简介：把去噪 latent 特征视为**时间函数**，用 **Chebyshev 多项式**（ridge regression 拟合）做全局长程近似——超越 Taylor 的局部外推限制，误差有紧致上界。FLUX 上 **4.79× 加速**。

* **LESA** (2026-02)：
  * 论文：[arXiv 2602.20497](https://arxiv.org/abs/2602.20497)
  * 简介：两阶段训练框架，用 **Kolmogorov-Arnold Networks (KAN)** 学习时序特征映射。多阶段多专家架构为不同噪声水平分配专门预测器。可达 **6.25× 加速**。

* **PrediT** (Predict to Skip, 2026-02)：
  * 论文：[arXiv 2602.18093](https://arxiv.org/abs/2602.18093)
  * 简介：把特征预测形式化为**线性多步问题**，用 Adams-Moulton 式方法从历史信息预测未来输出。高动态区域配 corrector，加动态步长调制。**5.54× 延迟削减**。

* **L2P-Cache** (Learnable Linear Predictor, CVPR 2026)：
  * 地址：https://github.com/Aredstone/L2P-Cache ![](https://img.shields.io/github/stars/Aredstone/L2P-Cache.svg)
  * 论文：[CVPR 2026 / arXiv 2604.26365](https://arxiv.org/abs/2604.26365)
  * 简介：把固定 Taylor / Adams-Bashforth 系数替换成**可学习的 per-timestep 线性权重**，从历史 feature trajectory 重建当前特征。单卡约 20 秒快速训练，FLUX.1-dev 上 4.55× FLOPs reduction / 4.15× latency speedup，Qwen-Image 上最高 7.18×。

* **LearniBridge** (ICML 2026)：
  * 地址：https://github.com/Iiiiiiirene/LearniBridge ![](https://img.shields.io/github/stars/Iiiiiiirene/LearniBridge.svg)
  * 论文：[ICML 2026 / arXiv 2606.26778](https://arxiv.org/abs/2606.26778)
  * 简介：发现高加速比下 cache 误差的最优校准更新落在跨 prompt 共享的低秩子空间，提出轻量 **LoRA bridge** 在多个 timestep 间校准 feature cache。只需 3–5 个训练样本，FLUX / HunyuanVideo / Wan2.1 分别达到 5.87× / 5.75× / 4.10×。

* **RFC** (Relational Feature Caching, ICLR 2026)：
  * 论文：ICLR 2026 Poster
  * 简介：利用输入-输出**关系**增强特征预测：Relational Feature Estimation (RFE) 用输入特征估计输出变化幅度，Relational Cache Scheduling (RCS) 仅在预测误差大时触发全量计算。

* **FeatFix** (2026-07)：
  * 论文：[arXiv 2607.27842](https://arxiv.org/abs/2607.27842)
  * 简介：抓住既有预测类方法的一个"浪费"：SpeCa 这类方法为了控制 draft drift 会**实算一个精确 block 特征做验证**，但这个特征只被用来量误差或指导决策，随后就丢掉了。FeatFix 指出它可以直接用于**校正**——在验证站点把 draft block 输出整块替换成同一入态算出的精确输出，从而把局部 draft 残差归零、削减下游误差。刻意不做 token / channel 级部分替换，也不做整步重算，只在**固定稀疏的 layer–timestep 站点**生效。四个图像 / 视频 backbone 上最高 **6.70×**。

* **RACER** (Disagree to Accelerate, 2026-08)：
  * 地址：https://github.com/LiZaiyuan0619/RACER ![](https://img.shields.io/github/stars/LiZaiyuan0619/RACER.svg)
  * 论文：[arXiv 2608.01740](https://arxiv.org/abs/2608.01740)
  * 简介：把问题从"怎么预测得更准"换成"**该信这个预测多少**"。核心观察：两路 forecast 在特征轨迹平滑处会一致、在难预测处会分歧，所以**分歧度本身就是免费的运行时可靠性信号**（不需要额外一次 denoiser 评估）。RACER 据此做两件事：把不确定的 forecast 向最近一次实算特征**收缩**（有确定性误差界），并在最危险的步**刷新**、再通过跳过后续一个已排定的实算来"偿还"这次开销 —— 因此是 closed-loop 而非 open-loop cache。等 NFE 下在 SD3.5-Large / FLUX.1-dev / Wan2.1-14B / HunyuanVideo（DrawBench / VBench / COCO）上全面优于最强 open-loop baseline；套在较弱的 Taylor 基底上也能把掉的质量捞回大半，说明它与具体 forecaster 设计解耦。

### 3.5 Fine-Grained / Granularity（token / region / channel）

在 **token / region / channel** 维度决定哪些局部单元重算、复用旧状态或共享 partial result。

* **ToCa (Token-wise Feature Caching)**：
  * 地址：https://github.com/Shenyi-Z/ToCa ![](https://img.shields.io/github/stars/Shenyi-Z/ToCa.svg)
  * 论文：[ICLR 2025 / arXiv 2410.05317](https://arxiv.org/abs/2410.05317)
  * 简介：首次在 **token 粒度**研究 DiT cache。发现不同 token 对缓存敏感度显著不同，细粒度选择适合 cache 的 token。

* **DuCa (Dual Feature Cache)**：
  * 地址：https://github.com/Shenyi-Z/DuCa ![](https://img.shields.io/github/stars/Shenyi-Z/DuCa.svg)
  * 论文：[arXiv 2412.18911](https://arxiv.org/abs/2412.18911)
  * 简介：ToCa 升级，token × layer 双层缓存。

* **FastCache**：
  * 地址：https://github.com/NoakLiu/FastCache-xDiT ![](https://img.shields.io/github/stars/NoakLiu/FastCache-xDiT.svg)
  * 论文：[arXiv 2505.20353](https://arxiv.org/abs/2505.20353)
  * 简介：静态 token 用**可学习线性近似**直接映射，活跃 token 全算，可达 4.5× 激进加速。

* **DiCache**：
  * 地址：https://github.com/Bujiazi/DiCache ![](https://img.shields.io/github/stars/Bujiazi/DiCache.svg)
  * 论文：[arXiv 2508.17356](https://arxiv.org/abs/2508.17356)
  * 简介：**让模型自己决定 cache**——用 shallow feature 作为 probe，基于变化触发重算。

* **ClusCa (ACM MM 2025)**：
  * 地址：https://github.com/Shenyi-Z/Cache4Diffusion ![](https://img.shields.io/github/stars/Shenyi-Z/Cache4Diffusion.svg)（与 SpeCa / TaylorSeer 共用的统一仓库）
  * 论文：[arXiv 2509.10312](https://arxiv.org/abs/2509.10312)
  * 简介：在 fresh step 用 **KMeans 对 token 聚类**，后续步每簇只算 1 个 token，其余簇内传播。token 计算量降 >90%，在 FLUX 上 **4.96× 加速**，ImageReward 保持在原版 99.49%。论文标题"Compute Only 16 Tokens in One Timestep"信息量很大。

* **HetCache** (2026-03)：
  * 论文：[arXiv 2603.24260](https://arxiv.org/abs/2603.24260)
  * 简介：面向 masked V2V 视频编辑，把 spatio-temporal token 显式拆成 **context** 与 **generative** 两类，仅缓存与 generative token 关联最强、语义最具代表性的 context token，跳过其余冗余注意力。2.67× 加速且编辑保真度近无损。

* **AccelAes** (2026-03)：
  * 地址：https://github.com/xuanhuayin/AccelAes ![](https://img.shields.io/github/stars/xuanhuayin/AccelAes.svg)
  * 论文：[arXiv 2603.12575](https://arxiv.org/abs/2603.12575)
  * 简介：观察到 denoising 在**美学描述词**对应的空间位置上是非均匀的——cross-attention 的高响应区才是真正影响美学评分的区域。AccelAes 据此对低相关区域做 spatio-temporal 削减，把算力集中到美学敏感区。Lumina-Next 上 2.11× 加速且 **ImageReward +11.9%**（罕见的"加速反提质"案例）。

* **JiT** (Just-in-Time, CVPR 2026)：
  * 地址：https://github.com/Wenhao-Sun77/Just-in-Time ![](https://img.shields.io/github/stars/Wenhao-Sun77/Just-in-Time.svg)
  * 论文：[CVPR 2026 / arXiv 2603.10744](https://arxiv.org/abs/2603.10744)
  * 简介：首次聚焦**空间冗余**而非时序冗余。构建由动态选择的稀疏 anchor token 驱动的 **spatially approximated generative ODE**——只算 token 子集，其余通过确定性 micro-flow（有限时间 ODE）保持结构一致性。FLUX.1-dev 上 **7× 加速**，是当前 token 类方法的最强单点。

* **ToPi** (2026-02)：
  * 论文：[arXiv 2602.01609](https://arxiv.org/abs/2602.01609)
  * 简介：面向**in-context 生成**（如参考图拼接）的 training-free token 剪枝。通过离线校准的敏感度分析识别关键 attention 层，derive 影响力指标做 context token 选择性剪枝 + 时序更新策略。>30% 加速。

* **TAP** (Token-Adaptive Predictor, 2026-03)：
  * 论文：[arXiv 2603.03792](https://arxiv.org/abs/2603.03792)
  * 简介：用首层全量计算作低成本 probe，为候选预测器（不同阶数 / 不同 horizon 的 Taylor 展开）计算 proxy loss，然后**每个 token 自选 proxy loss 最小的预测器**——异构 per-token 预测选择。

* **MoECa** (2026-06)：
  * 论文：[arXiv 2606.15615](https://arxiv.org/abs/2606.15615)
  * 简介：面向 **DiT-MoE**（Mixture-of-Experts DiT）的细粒度缓存。在 expert branch 级做跨 timestep 特征复用，引入 expert-aware 自适应控制和 MoE / attention 路径同步缓存更新，保持中间状态稳定。

* **Kaleido** (2026-07)：
  * 论文：[arXiv 2607.13770](https://arxiv.org/abs/2607.13770)
  * 简介：利用 RoPE 后 token channel 对时间 / 水平 / 垂直方向的结构化相关性，在线比较邻接 token；高相似 channel 完整复用前一个 token 的 partial attention / GEMM 结果，中等相似 channel 只补低位乘法。配套可重构 systolic-array PE 与 data dispatcher 处理不规则 reuse pattern。在 HunyuanVideo / Wan2.1 / CogVideoX / TurboDiffusion 上，16nm RTL / cycle-level 仿真相对现有专用加速器最高 **5.9×**，energy-saving ratio 最高 **16.0×**。这是**同一步内的局部结果复用**，不是传统 cross-timestep cache。

* **OmniCache** (2026-07)：
  * 论文：[arXiv 2607.23844](https://arxiv.org/abs/2607.23844)
  * 简介：把冗余来源系统归成四类 —— **intra-frame、inter-frame、motion、denoising-step**，对应 Token Cache / Frame Cache / Block Cache / Layered Cache 四级统一分层框架。与 token-merging 系方法的关键差别是**不做特征平均**：用相似度匹配挑出可缓存项、跳过其计算，再把缓存激活**按原位置恢复**，保住特征顺序与时空结构；并让空间特征在时间层复用、时间特征在空间层复用。training-free，SD3 / SVD-XT / Latte 上 latency 分别降 35% / 25% / 28%。

* **HeadCast** (2026-07)：
  * 地址：https://github.com/sjlgaga/HeadCast ![](https://img.shields.io/github/stars/sjlgaga/HeadCast.svg)
  * 论文：[arXiv 2607.20125](https://arxiv.org/abs/2607.20125)
  * 简介：面向 **AR 视频扩散**的 KV cache 侧优化。AR 长视频里 KV cache 持续增长、attention 成为主要开销，而已有 eviction 启发式太粗会导致帧间闪烁。HeadCast 发现预训练 AR 模型的 attention head 行为**稳定且异质**，于是在最大噪声步做一次性分类，把每个 head 归入 **Sink / Dummy / Spatial / Global** 四种原型，并把单块 KV cache 重构成 head-specific 通路 —— 关键是**保留 Global head**，因为激进 eviction 恰恰破坏的是它们承载的长程时序一致性。Spatial 通路跑在固定尺寸网格上，所以收益随分辨率上升：720P 1.62×、1080P 1.95×，VBench 与 full attention 相当且基本无闪烁。training-free、plug-and-play。

* **DSTAR** → 见 §5（sparse attention reuse + 差分激活混合精度 + 专用加速器）。
* **DiTango** → 见 3.10（Context-Parallel 下按通信拓扑决定 attention state 复用）。
* **FlashDiff** → 见 3.10（semantic region 跨 timestep 复用 prior state，并联动多请求调度）。

### 3.6 Frequency-Aware（频域类）

在**频率维度**区分高低频特征的不同时序行为。

* **FEB-Cache**：
  * 地址：https://github.com/aSleepyTree/EB-Cache ![](https://img.shields.io/github/stars/aSleepyTree/EB-Cache.svg)
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
  * 论文：[arXiv 2603.05315](https://arxiv.org/abs/2603.05315)
  * 简介：三轴正交的 Hybrid 频域 cache：
    - **TADS** (Timestep-Aware Dynamic Scheduling)：cosine bell 时步阈值调度
    - **CEB** (Cumulative Error Budget)：连续缓存上限 C_max，防误差级联
    - **FDC** (Frequency-Decomposed Caching)：高低频带**非对称阈值**（低频严 γ=0.8 / 高频松 γ=1.5）
  * 在 FLUX.1-schnell 上 **2.46× 加速**，比 TeaCache 快 16%。

* **E²-CRF** (2026-04)：
  * 论文：[arXiv 2604.22901](https://arxiv.org/abs/2604.22901)
  * 简介：面向频域扩散模型。利用**频谱局部化**（能量集中于低频）和**镜像对称**将有效频域维度减半；构建**闭环误差反馈系统**，事件驱动地缓存 transformer KV 特征并按残差动力学触发重算，替代固定调度。~2.2× 加速。

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
* **BlockDance** → 见 3.3（DiT-XL/2 / PixArt-α / Open-Sora 通吃）
* **ProfilingDiT** → 见 3.3（Wan2.1 专用，FG/BG 拆分）
* **ERTACache** → 见 3.2（Wan2.1 上 VBench 几乎无损）
* **MixCache** (Mixture-of-Cache)：
  * 论文：[arXiv 2508.12691](https://arxiv.org/abs/2508.12691)
  * 简介：多个 cache 策略组成 mixture，router 动态选择。
* **BWCache** (Block-Wise Cache)：
  * 地址：https://github.com/hsc113/BWCache ![](https://img.shields.io/github/stars/hsc113/BWCache.svg)
  * 论文：[arXiv 2509.13789](https://arxiv.org/abs/2509.13789)
  * 简介：视频 DiT 的 block-wise 缓存。
* **EasyCache** → 见 3.2
* **HetCache** → 见 3.5（MV2V 编辑专用，token 异质拆分）
* **DisCa** → 见 3.4（蒸馏后视频 DiT，11.8×）
* **WorldCache** (2026-03)：
  * 地址：https://github.com/umair1221/WorldCache ![](https://img.shields.io/github/stars/umair1221/WorldCache.svg)
  * 论文：[arXiv 2603.22286](https://arxiv.org/abs/2603.22286)
  * 简介：首个专攻**视频 world model**（如 Cosmos-Predict2.5-2B）的 cache 工作。常规 cache 假设特征近静态，在动态场景会产生 ghosting / blur。WorldCache 引入**motion-adaptive 阈值** + saliency-weighted drift 估计 + blending&warping 近似 + phase-aware 阈值调度，在 2.3× 加速下保留 99.4% 基线质量。

* **FlowCache** (2026-02)：
  * 地址：https://github.com/mikeallen39/FlowCache ![](https://img.shields.io/github/stars/mikeallen39/FlowCache.svg)
  * 论文：[arXiv 2602.10825](https://arxiv.org/abs/2602.10825)
  * 简介：面向**自回归视频** chunk-by-chunk 生成场景。发现不同 chunk 去噪模式差异大，统一 cache 不优；提出 chunkwise 独立 cache 策略 + importance-redundancy 联合 KV 压缩。MAGI-1 上 2.38×，**SkyReels-V2 上 6.7×**。

* **X-Cache** (2026-04)：
  * 论文：[arXiv 2604.20289](https://arxiv.org/abs/2604.20289)
  * 简介：面向**自回归世界模型**（多相机驾驶 world model X-world，基于多 block causal DiT + 滚动 KV cache 的 few-step 去噪）。提出**跨 chunk** 而非跨 step 的 residual 复用：用 structure / action-aware 的 block-input fingerprint 做 dual-metric gating 决定 recompute or reuse，并强制 KV update chunk 全算以阻断 AR 误差累积。生产级模型上 **2.6× 加速**且性能近无损。

* **FIS-DiT** (2026-05)：
  * 论文：[arXiv 2605.11869](https://arxiv.org/abs/2605.11869)
  * 简介：把优化轴从"时序去噪轨迹"换到"**latent 帧维度**"——观察到 few-step 视频 DiT 在帧位置上存在**结构一致性**与帧维稀疏性，于是在稳定 block 上稀疏计算**帧子集 latent slice**、敏感 block 上保留全计算，**完全不引入额外 feature-cache 显存**。Wan 2.2 / HunyuanVideo 1.5 上 2.11–2.41× 加速，VBench-Q / CLIP 近无损。

* **PreciseCache** (ICLR 2026)：
  * 论文：[ICLR 2026 / arXiv 2603.00976](https://arxiv.org/abs/2603.00976)
  * 简介：自适应视频加速框架：**LFCache**（step-wise）+ **BlockCache**（block-wise）双层精准识别冗余特征并跳过。Wan2.1-14B 上约 **2.6× 加速**，无明显质量降损。

* **SCOPE** (2026-04)：
  * 论文：[arXiv 2604.02979](https://arxiv.org/abs/2604.02979)
  * 简介：面向 AR 视频扩散的 training-free 框架，引入**三模调度器**（cache / predict / recompute）——预测模式用噪声级 Taylor 外推填补复用与重算间的空白；选择性计算把执行限定在 active 帧区间。MAGI-1 / SkyReels-V2 上 **4.73× 加速**。

* **MotionCache** (2026-05)：
  * 论文：[arXiv 2605.01725](https://arxiv.org/abs/2605.01725)
  * 简介：利用帧间差分作为像素级运动特征的轻量代理，**按运动幅度动态调整去噪步频率**——静态像素区域激进跳步，高运动区域精算。

* **HSA** (Heterogeneous Step Allocation, 2026-05)：
  * 项目：https://ernestchu.github.io/hsa
  * 论文：[arXiv 2605.06892](https://arxiv.org/abs/2605.06892)
  * 简介：观察视频 DiT 中不同 spatiotemporal token 不需要同样多的 denoising steps，按 velocity dynamics 给 token 分配异构 step budget。为解决序列长度不一致，引入 **KV-cache synchronization** 让 active token 仍可 attend 到全序列，并用 cached Euler update 一步推进 skipped token。Wan-2 / LTX-2 上在 50% / 25% runtime 等激进预算下保持更优 Pareto。

* **SyncCache** (ECCV 2026)：
  * 论文：[ECCV 2026 / arXiv 2606.30849](https://arxiv.org/abs/2606.30849)
  * 简介：面向 audio-driven portrait animation 的 training-free cache。利用音频驱动高频人脸区域与低频背景的非对称动态，做 **Spatially-Asymmetric Probing** 与 **Modality-Decoupled Caching**：heavy DiT block 的稳定 inter-block residual 复用，轻量 audio block 持续重算以保 lip sync。HunyuanVideo-Avatar 4.12×、Wan-S2V 3.75×。

* **WorldDynCache** (2026-08)：
  * 论文：[arXiv 2608.01845](https://arxiv.org/abs/2608.01845)
  * 简介：面向**扩散 world model**。指出既有 cache 的判据（局部 drift 或短程原生空间历史）会漏掉两件事：一是跨被跳过的步**累积**的 latent transition 近似缺陷，二是**相位 / 条件依赖**的 latent 演化方向变化。方案两件：轻量 **latent-transition 风险估计器**追踪近似缺陷的未来累积影响，并在 exact anchor 处用反事实缺陷校准自己的预测；condition- 与 phase-aware 的 **lifted latent surrogate** 在不额外跑 transformer 的前提下近似 latent 演化。HunyuanVoyager-13B **4.92×**、Aether-5B **2.15×**，且在 WorldScore / PSNR / SSIM / LPIPS 上是所比 cache 方法里质量最好的。与 §3.8 的 WorldCache 是同一战场的两种思路（motion-adaptive 阈值 vs. 风险受控 latent 动力学）。

* **EchoCache** (ACM MM 2026)：
  * 地址：论文声明 https://github.com/IF-LAB-PKU/EchoCache（核验于 2026-08-08 仍未公开）
  * 论文：[ACM MM 2026 / arXiv 2608.02474](https://arxiv.org/abs/2608.02474)
  * 简介：面向 **audio-driven video generation (A2V)**。既有 cache 只挖视觉特征的时序冗余，忽略了 A2V 的跨模态特性 —— 音频驱动视觉、且其时间重要性高度非均匀。论文点出两层错配：**temporal-semantic** 与 **computation-storage**。EchoCache 用**音频时频能量**作 saliency anchor 引导 latent 级缓存更新，再加 dynamic timestep-latent 缓存机制与量化 cache 管理兼顾效率和显存。Wan2.2-S2V + EMTD 上 **2.46×** 且综合最优。与 SyncCache（音频驱动人像、模态解耦 residual）是同一模态、不同切入点。

* **HeadCast** → 见 3.5（AR 视频 KV cache 的 head 级原型分通路，720P 1.62× / 1080P 1.95×）
* **CachedSearch** → 见 3.9（视频 test-time search 中用 cache 换探索宽度）
* **ACID** → 见 3.2（适配 HunyuanVideo / Wan2.1 / CogVideoX 的 critical-step 双阈值 wrapper）
* **Kaleido** → 见 3.5（视频 DiT 的 channel-wise partial-result reuse + 专用硬件）
* **FlashDiff** → 见 3.10（视频与图像/音频统一的 semantic region reuse + serving scheduler）
* **CODA** → 见 3.9 / §5（边缘视频 DiT 的 compute-cache operator disaggregation）
* **DSTAR** → 见 §5（覆盖图像 / 视频 / 编辑七类 DiT 的时空冗余削减 + 加速器）

### 3.9 Hybrid / Multi-Dimensional（混合类）

组合多个轴（time × layer × frequency × CFG × token / region / channel × system）的混合方法。

* **FasterCache**：
  * 地址：https://github.com/Vchitect/FasterCache ![](https://img.shields.io/github/stars/Vchitect/FasterCache.svg)
  * 论文：[ICLR 2025 / arXiv 2410.19355](https://arxiv.org/abs/2410.19355)
  * 简介：Feature × CFG × Frequency 三轴融合，Vchitect-2.0 上 1.67×。
* **HyCa** → 见 3.4（per-dim ODE 混合，算 Predictive × Residual 双轴 hybrid）
* **ERTACache** → 见 3.2（Timestep × Residual 两类误差联合矫正）
* **SpectralCache** → 见 3.6
* **LayerCache** → 见 3.3（Layer + Predictive 两轴）
* **SODA** → 见 3.3（敏感度 + DP 联合 caching/pruning）
* **MoECa** → 见 3.5（MoE branch 级跨步复用）
* **HSA** → 见 3.8（Token step budget + KV-cache synchronization）
* **LearniBridge** → 见 3.4（Feature cache + low-rank LoRA calibration）
* **SyncCache** → 见 3.8（Spatial + modality + residual cache）
* **FlashDiff** → 见 3.10（Region × Timestep × Serving）
* **Kaleido** → 见 3.5（Token × Channel × Hardware）
* **OmniCache** → 见 3.5（Token × Frame × Block × Layer 四级分层）
* **EchoCache** → 见 3.8（Audio energy × Timestep × Latent，含量化 cache 管理）
* **WorldDynCache** → 见 3.8（Risk × Phase/Condition × Latent surrogate）
* **CachedSearch** (2026-07)：
  * 论文：[arXiv 2607.23159](https://arxiv.org/abs/2607.23159)
  * 简介：把 cache 用在一个此前没人碰的轴上 —— **test-time search 的预算分配**。动机：video test-time search 让小模型能追上大模型，但代价是 2–10× 算力，且所有候选都被完整去噪、绝大多数最后被丢弃。论文先做了第一份系统研究回答"**有损 cache 会不会破坏 verifier 的候选排序**"：Wan2.1-T2V-1.3B 上带 ~2× 自适应 cache 的 rollout 与全量 rollout，逐 prompt Spearman 秩相关中位数 **0.905**、VBench top-1 一致率 **72%**，且误差集中在本就接近平手的候选之间 —— 所以排序损坏是**自限的**。据此 CachedSearch 用激进 cache 探索全部候选，只把胜者以全算力重新生成一遍：N=8 时以 **63% 成本**拿到 best-of-N 的 **94.7%** 收益；等预算下可搜两倍宽度、多拿 38% 收益；配合中途剪枝，探索侧节省放大到 **3.11×**（保留 88.6% 收益）。在 Wan / LTX / CogVideoX / Hunyuan 四个家族、1.3B–14B 六个模型上成立，且 training-free、verifier-agnostic、与具体搜索算法正交。

* **EVO** (Evolving Cache Schedules, PRCV 2026)：
  * 地址：https://github.com/pillom/EVO ![](https://img.shields.io/github/stars/pillom/EVO.svg)
  * 论文：[PRCV 2026 / arXiv 2607.20293](https://arxiv.org/abs/2607.20293)
  * 简介：把 cache 带到 **diffusion policy（视觉运动控制）** 这个新场景 —— 这里的约束不是图像质量而是**闭环 rollout 成功率**，且实时性要求更硬。指出既有 training-free schedule 在各 block 上均匀分配算力，忽略 block 间冗余异质。EVO 把每个候选表示为 block–timestep 格点上的**完整 schedule**，用进化搜索做全局优化；为让搜索可行，引入 redundancy-aware 初始化播种优质个体、target-conditioned early stopping 达标即停。离线搜出的 schedule 可直接插到预训练 policy 上、无需重训。多个操作 benchmark 上 action generation 最高 **8.05×**，FLOPs 从 15.77G 降到 1.96G。思路上与 ECAD（图像 DiT 的进化 cache 搜索）同源，场景与目标函数不同。

* **CODA** (MICRO 2026)：
  * 论文：[MICRO 2026 / arXiv 2607.14908](https://arxiv.org/abs/2607.14908)
  * 简介：面向显存受限的 edge VDM，把 compute-intensive dense path 留在 xPU，把 memory-bound cross-timestep cache path 重组为 coarse-grained segment 并下沉到轻量 **DIMM-NMP**；再利用 CFG 两分支独立性，把一侧 cache DMA / NMP 与另一侧 dense compute 流水重叠。RTX 4090 profiling + Ramulator / NMP RTL 建模在 Latte / Open-Sora / Wan2.1 / HunyuanVideo / CogVideoX 等模型上给出最高 **1.80×** 端到端加速、**1.74×** 能效提升；这些是协同建模结果，并非 NMP 实芯片测量。

### 3.10 Service-Level Cache（跨请求 / 区域调度）

服务层有两条正交路线：一条在**历史请求之间**直接复用 feature；另一条仍在单请求内做局部 cache，但把省下来的算力动态分配给并发请求。它们关注的不只是单样本 FLOPs，还包括端到端 latency、throughput、负载均衡与通信开销。

* **Chorus** (2026-04)：
  * 论文：[arXiv 2604.04451](https://arxiv.org/abs/2604.04451)
  * 简介：首次系统化研究 4-step 蒸馏视频 DiT 的**跨请求 cache 复用**：denoising 过程分三段——早期 full reuse、中段 region-specific cache、后段刷新；配套 **Token-Guided Attention Amplification** 维持 prompt 语义对齐。在工业级 4-step 模型上达 **45% 加速**（≈1.45×）。

* **FlashDiff** (2026-07)：
  * 论文：[arXiv 2607.12121](https://arxiv.org/abs/2607.12121)
  * 简介：用 warm-up 阶段的 cross-attention 把 latent 切成语义一致的 image/video patch 或 audio segment，runtime controller 判断 region 是否已稳定；被跳过的 region 直接复用相邻 timestep 的 prior state。服务端再以 affinity-aware scheduler 把释放的 compute slack 分配给并发请求。覆盖 SD3 / FLUX / Wan2.1 / Stable Audio Open，减少 **24–66%** 计算，在线 request completion time（含排队/调度收益）降 **30–97%**、throughput 提升 **1.2–2.2×**。它不跨请求共享 feature，与 Chorus 的复用边界不同。

* **DiTango** (2026-07)：
  * 论文：[arXiv 2607.15650](https://arxiv.org/abs/2607.15650)
  * 简介：服务层的第三条路线 —— 把复用决策与**多机通信拓扑**绑定。并行推理（Context Parallelism）在多节点下的瓶颈是通信开销，而论文观察到 CP 的 sequence partition 存在明显异质：**空间上邻近的 partition 对 attention 结果贡献更大**。把这个异质模式映射到分层通信拓扑，就能以更低通信代价优先访问高贡献 partition。DiTango 据此提出 selective attention state 机制，在"部分 attention 实算"与"跨去噪步复用历史结果"之间做权衡：anchor-guided state selection planner 为每个 partition 定 compute-or-reuse 决策，配套 runtime 编排 state-centric 操作。多节点下 **1.9×** 端到端、**3.2×** attention 加速且近线性扩展，质量与 SOTA 相当。注意它的加速比是并行系统口径，不能与单卡算法 latency 直接横比。

## 4. 测评

### 4.1 常用评测指标

| 类别 | 指标 | 含义 |
|------|------|------|
| 加速 | **Speedup** | 相对无 cache 的 wall-clock 加速比 |
| 加速 | **Latency / step** | 单步推理时延 |
| 服务 | **Request Completion Time (RCT) ↓** | 含排队、调度与执行的请求完成时间 |
| 服务 | **Throughput ↑** | 单位时间完成的请求数 |
| 计算 | **FLOPs / Compute Reduction ↑** | 被 cache / reuse 消除的理论或实测计算量 |
| 能效 | **Energy Saving / Efficiency ↑** | 基线能耗与方案能耗之比，或单位能量吞吐收益 |
| 像素级 | **PSNR ↑** | 峰值信噪比（vs. 无 cache 输出）|
| 结构级 | **SSIM ↑** | 结构相似度 |
| 感知级 | **LPIPS ↓** | 感知距离（AlexNet/VGG）|
| 分布级 | **FID ↓** | Frechet Inception Distance |
| 文图对齐 | **CLIP-Score ↑** | text-image 对齐 |
| 视频时序 | **VBench** | 视频质量多维度评测 |
| 视频时序 | **Temporal Flickering / Motion Smoothness** | 时序连贯性 |
| 音频 | **FD / KL ↓, CLAP ↑** | 音频分布质量、多样性与文本-音频对齐 |

### 4.2 基线模型

**图像**：SD 1.5 / SDXL / PixArt-α / PixArt-Σ / FLUX.1-dev / FLUX.1-schnell / SD3 / Qwen-Image / Z-Image / LongCat-Image / Lumina-Next / DiT-MoE

**视频**：Open-Sora / Open-Sora-Plan / Latte / CogVideoX / HunyuanVideo / HunyuanVideo-Avatar / Wan2.1 / Wan2.2 / Wan-S2V / LTX-Video / LTX-2 / Mochi / Vchitect-2.0 / TurboDiffusion / MAGI-1 / SkyReels-V2 / Cosmos-Predict2.5

**音频**：Stable Audio Open

### 4.3 常用 Benchmark

| Benchmark | 用途 | 链接 |
|-----------|------|------|
| **COCO-30K** | 图像 FID / CLIP 评测 | - |
| **MJHQ-30K** | 图像质量评测 | - |
| **GenEval** | 图像文图对齐评测 | [djghosh13/geneval](https://github.com/djghosh13/geneval) |
| **DPG-Bench** | 长文图对齐 | [TencentQQGYLab/ELLA](https://github.com/TencentQQGYLab/ELLA) |
| **VBench** | 视频生成 16 维度评测 | [Vchitect/VBench](https://github.com/Vchitect/VBench) |
| **OneIG-Bench** | 统一图像生成评测 | - |
| **AudioCaps** | 文生音频 FD / KL / CLAP 评测 | [audiocaps.github.io](https://audiocaps.github.io/) |

## 5. 工程、系统与硬件

| 工程 / 系统 / 硬件 | 类型 | 说明 |
|--------------------|------|------|
| **xFuser / ParaAttention** | 开源框架 | 并行 + cache 一体化框架，TeaCache / FBCache / SpectralCache 的集成入口 |
| **VideoSys** | 开源框架 | 视频 DiT 推理优化框架，PAB 官方实现载体 |
| **Diffusers pipeline hooks** | 接入机制 | 通过 hook 注入 cache 的通用模式 |
| **vLLM-Omni Diffusion Cache** | 服务框架 | vLLM 引入的 diffusion cache 工程化实现 |
| **TensorRT-LLM / TensorRT** | 部署框架 | cache + low-precision 联合部署 |
| **[FlashDiff](https://arxiv.org/abs/2607.12121)** | 服务系统（未开源） | semantic patch execution + affinity-aware scheduler；把 region cache 节省转化为多请求 RCT / throughput 收益 |
| **[Kaleido](https://arxiv.org/abs/2607.13770)** | 算法-硬件协同（未开源） | channel-wise partial-result reuse + 可重构 systolic-array PE / data dispatcher（16nm RTL / cycle-level 仿真）|
| **[CODA](https://arxiv.org/abs/2607.14908)** | 算法-硬件协同（未开源） | xPU / DIMM-NMP compute-cache operator disaggregation + CFG-interleaved pipeline（profiling + NMP 建模）|
| **[DSTAR](https://arxiv.org/abs/2607.15846)** | 算法-硬件协同（未开源，MICRO 2026） | 空间 + 时间冗余联合削减：差分激活细粒度混合精度量化 + sparse attention reuse + 专用加速器；7 类 DiT 上 vs. A100 最高 7.33× / 41.89× energy，vs. SOTA accelerator 2.54× / 3.68×，无精度损失 |
| **[DiTango](https://arxiv.org/abs/2607.15650)** | 并行推理系统（未开源） | Context-Parallel × selective attention state reuse；anchor-guided planner + state-centric runtime，多节点 1.9× 端到端 / 3.2× attention，近线性扩展 |

## 6. 相关综述

* **A Survey on Cache Methods in Diffusion Models: Toward Efficient Multi-Modal Generation** ([arXiv 2510.19755](https://arxiv.org/abs/2510.19755)) — 2025 年 10 月，目前最全最新的 cache 综述，将方法分为 static / timestep-adaptive / layer-adaptive / predictive / hybrid 五大类。
* **Efficient Diffusion Models: A Comprehensive Survey** — 覆盖量化、蒸馏、cache、并行等全方向加速。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=leeguandong/Awesome-Dit-Cache&type=Date)](https://star-history.com/#leeguandong/Awesome-Dit-Cache&Date)

## License

[Apache License 2.0](LICENSE)
