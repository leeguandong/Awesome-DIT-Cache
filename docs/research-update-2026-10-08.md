# 论文检索与更新记录：2026-10-08

本次检查以仓库上次提交（2026-08-08）为起点，检索截至 2026-10-08 可访问的公开论文；README 新增 28 篇。标题、提交日期与机制以 arXiv 官方 API / 全文为准，代码状态以 GitHub 公共 API 与文件树为准。会议标签只在作者明确注明接收时使用，未独立核验会议 proceedings。未运行论文实现或复现加速比。

检索使用 arXiv API 的 `diffusion AND (cache OR caching OR reuse)`，按提交时间降序查看前 200 条；另补查 `diffusion / DiT / flow matching AND feature forecasting / feature prediction / feature reuse`，日期限定 2026-08-08 至 2026-10-08。此为本轮检索覆盖范围，不声称穷尽所有未使用这些关键词的工作。

日期采用 API `published`，不从 arXiv 编号推断；例如 Spectral-Guided Diffusion 的编号为 2609.29505，但 API 首次提交日期为 2026-08-24。

## 新增论文与核验结果

| 方法 / 官方完整标题 | 首次提交 | arXiv | 粒度 / 策略章节 | 代码状态（2026-10-08） |
|---|---|---|---|---|
| **BAG** — BAG: Budget-Aware Gating for Diffusion Caching | 2026-08-10 | [2608.09231](https://arxiv.org/abs/2608.09231) · [全文](https://arxiv.org/html/2608.09231) | §2.1 / §3.2 | [Westlake-AGI-Lab/BAG](https://github.com/Westlake-AGI-Lab/BAG)：公开项目仓库；尚未发现实现文件 |
| **GCache** — From Local Mismatch to Global Impact: Optimizing Cache Reuse Policy for Efficient Diffusion | 2026-08-13 | [2608.13043](https://arxiv.org/abs/2608.13043) · [全文](https://arxiv.org/html/2608.13043) | §2.1 / §3.2 | 论文全文未找到代码链接 |
| **GeoCache** — GeoCache: Training-Free Acceleration of Multi-View Texture Diffusion via Geometric Delta Transport | 2026-08-13 | [2608.13255](https://arxiv.org/abs/2608.13255) · [全文](https://arxiv.org/html/2608.13255) | §2.5 / §3.5 | 论文全文未找到代码链接 |
| **DriveCache** — DriveCache: Action-Aware Caching for Driving World Model Inference | 2026-08-17 | [2608.16354](https://arxiv.org/abs/2608.16354) · [全文](https://arxiv.org/html/2608.16354) | §2.1 / §3.8 | 论文全文未找到代码链接 |
| **LinCa** — LinCa: Accelerating Diffusion Models via Learnable Decomposed Feature Caching | 2026-08-18 | [2608.17973](https://arxiv.org/abs/2608.17973) · [全文](https://arxiv.org/html/2608.17973) | §2.1 / §3.4 | [QHR69/LinCa](https://github.com/QHR69/LinCa)：公开实现文件；未执行复现 |
| **ChebBooster** — ChebBooster: A Training-Free Approach for Efficient Diffusion Transformer Inference via Chebyshev-Inspired Extrapolation | 2026-08-24 | [2608.23429](https://arxiv.org/abs/2608.23429) · [全文](https://arxiv.org/html/2608.23429) | §2.1 / §3.4 | [Kiramei/ChebBooster](https://github.com/Kiramei/ChebBooster)：公开实现文件；未执行复现 |
| **BaryCache** — Memory-Efficient Training-Free Acceleration of Diffusion Transformers with BaryCache | 2026-08-24 | [2608.28670](https://arxiv.org/abs/2608.28670) · [全文](https://arxiv.org/html/2608.28670) | §2.1 / §3.4 | [Kiramei/BaryCache](https://github.com/Kiramei/BaryCache)：公开实现文件；未执行复现 |
| **SCR / Spectral-Guided Diffusion** — Spectral-Guided Diffusion: Accelerating Inference via Static Spectral Layer Scheduling | 2026-08-24 | [2609.29505](https://arxiv.org/abs/2609.29505) · [全文](https://arxiv.org/html/2609.29505) | §2.2 / §3.1 | 论文全文未找到代码链接 |
| **DensityKV** — DensityKV: Density-Guided KV Cache Compression for Long Video Generation | 2026-08-28 | [2608.27922](https://arxiv.org/abs/2608.27922) · [全文](https://arxiv.org/html/2608.27922) | §2.3 / §3.5 | [ZhaoWQQ/DensityKV](https://github.com/ZhaoWQQ/DensityKV)：公开实现文件；未执行复现 |
| **EpaCache** — EpaCache: Error-Propagation-Aware Caching for Accelerating Diffusion-Based Visual Generation | 2026-08-29 | [2608.29264](https://arxiv.org/abs/2608.29264) · [全文](https://arxiv.org/html/2608.29264) | §2.1 / §3.2 | 论文全文未找到代码链接 |
| **RegionCache** — RegionCache: Semantic-Aware Region Reuse for Efficient Multi-Turn Image Generation | 2026-08-30 | [2608.29809](https://arxiv.org/abs/2608.29809) · [全文](https://arxiv.org/html/2608.29809) | §2.5 / §3.10 | [hebutBryant/RegionCache](https://github.com/hebutBryant/RegionCache)：公开实现文件；未执行复现 |
| **GP-Refiner** — Accelerating Diffusion Transformers with Gaussian Process Rectified Feature Cache | 2026-09-05 | [2609.05981](https://arxiv.org/abs/2609.05981) · [全文](https://arxiv.org/html/2609.05981) | §2.1 / §3.4 | [Aredstone/GP-Refiner](https://github.com/Aredstone/GP-Refiner)：公开空仓库；尚无实现 |
| **RefAdapt-DiT** — RefAdapt-DiT: Adaptive Joint Attention for Reference-Conditioned Diffusion Transformers | 2026-09-26 | [2609.32415](https://arxiv.org/abs/2609.32415) · [全文](https://arxiv.org/html/2609.32415) | §2.3 / §3.5 | 论文全文未找到代码链接 |
| **Carnator** — Carnator: Fast Text-to-Video Generation with Generation-Native Compatibility-Guided Cross-Request Reuse | 2026-09-26 | [2609.32420](https://arxiv.org/abs/2609.32420) · [全文](https://arxiv.org/html/2609.32420) | §2.5 / §3.10 | 论文全文未找到代码链接 |
| **FlashForward** — In-Flight KV Cache with Clean Anchors for Faster Autoregressive Video Diffusion | 2026-09-26 | [2609.32540](https://arxiv.org/abs/2609.32540) · [全文](https://arxiv.org/html/2609.32540) | §2.3 / §3.8 | 论文全文未找到代码链接 |
| **WAMachine** — Efficient World Action Model Inference with Adaptive Intermediate States | 2026-09-28 | [2609.34608](https://arxiv.org/abs/2609.34608) · [全文](https://arxiv.org/html/2609.34608) | §2.8 / §3.9 | [RSIScience/WAMachine](https://github.com/RSIScience/WAMachine)：作者已声明地址；公开访问 404 |
| **RA-CFGCache** — RA-CFGCache: From Branch-Level Criteria to Guided-Risk Control under Classifier-Free Guidance | 2026-09-29 | [2609.36433](https://arxiv.org/abs/2609.36433) · [全文](https://arxiv.org/html/2609.36433) | §2.7 / §3.7 | [yiming-l21/RA-CFGCache](https://github.com/yiming-l21/RA-CFGCache)：公开实现文件；未执行复现 |
| **ParaAnya** — ParaAnya: Accelerating Parallel Diffusion Sampling with Plug-and-Play Output Caching | 2026-09-29 | [2609.36522](https://arxiv.org/abs/2609.36522) · [全文](https://arxiv.org/html/2609.36522) | §2.1 / §3.10 | [XXIIIII/ParaAnya](https://github.com/XXIIIII/ParaAnya)：作者已声明地址；公开访问 404 |
| **DeCoPrune** — DeCoPrune: Efficient KV-Cache Pruning for Autoregressive Video Diffusion via Denoising Consistency | 2026-09-30 | [2609.39096](https://arxiv.org/abs/2609.39096) · [全文](https://arxiv.org/html/2609.39096) | §2.3 / §3.5 | [DeCoPrune/CMBench](https://github.com/DeCoPrune/CMBench)：公开实现文件；未执行复现 |
| **Golden Path Hypothesis (GPH)** — The Golden Path Hypothesis: Reusable Schedules in Diffusion Caching | 2026-09-30 | [2609.39343](https://arxiv.org/abs/2609.39343) · [全文](https://arxiv.org/html/2609.39343) | §2.1 / §3.1 | [nanguoyu/Golden-Path-Hypothesis](https://github.com/nanguoyu/Golden-Path-Hypothesis)：公开实现文件；未执行复现 |
| **SpectralCache (World Models)** — SpectralCache: Accelerating Diffusion-Based World Models via Spectral Feature Caching | 2026-10-02 | [2610.02660](https://arxiv.org/abs/2610.02660) · [全文](https://arxiv.org/html/2610.02660) | §2.1 / §3.4 | 论文全文未找到代码链接 |
| **AutoTarget** — Rethinking What to Cache in Few-Step Diffusion Transformers: Solver-Aware Target Selection | 2026-10-02 | [2610.03577](https://arxiv.org/abs/2610.03577) · [全文](https://arxiv.org/html/2610.03577) | §2.1 / §3.9 | [wali1024-offical/AutoTarget](https://github.com/wali1024-offical/AutoTarget)：公开实现文件；未执行复现 |
| **ManifoldCache** — ManifoldCache: Training-Free Diffusion Acceleration via Constraint Manifold Caching | 2026-10-03 | [2610.04510](https://arxiv.org/abs/2610.04510) · [全文](https://arxiv.org/html/2610.04510) | §2.2 / §3.3 | [prinshul/Manifoldcache](https://github.com/prinshul/Manifoldcache)：公开项目仓库；尚未发现实现文件 |
| **HybridFF** — Hybrid-Basis Feature Forecasting for Diffusion Sampling Acceleration | 2026-10-04 | [2610.05254](https://arxiv.org/abs/2610.05254) · [全文](https://arxiv.org/html/2610.05254) | §2.1 / §3.4 | 论文全文未找到代码链接 |
| **Unexpired Plan** — The Unexpired Plan: A Free Monitor for Accelerated Diffusion Policies | 2026-10-05 | [2610.05747](https://arxiv.org/abs/2610.05747) · [全文](https://arxiv.org/html/2610.05747) | §2.1 / §3.9 | [YiZhao-Jasper/unexpired-plan](https://github.com/YiZhao-Jasper/unexpired-plan)：公开实现文件；未执行复现 |
| **MC-Sparse** — MC-Sparse: Deconstructing and Closing the Dense-Sparse Attention Gap in Diffusion Transformers | 2026-10-05 | [2610.06801](https://arxiv.org/abs/2610.06801) · [全文](https://arxiv.org/html/2610.06801) | §2.3 / §3.5 | [dodododddo/mcsparse](https://github.com/dodododddo/mcsparse)：公开实现文件；未执行复现 |
| **Koopman Observers** — Koopman Observers for Diffusion Acceleration: Correcting Feature Forecasts with Shallow Measurements | 2026-10-07 | [2610.10366](https://arxiv.org/abs/2610.10366) · [全文](https://arxiv.org/html/2610.10366) | §2.2 / §3.4 | 论文全文未找到代码链接 |
| **MORCA** — MORCA: Offline-to-Online Reinforcement Learning for Adaptive Cache Reuse in Video Diffusion Acceleration | 2026-10-07 | [2610.10457](https://arxiv.org/abs/2610.10457) · [全文](https://arxiv.org/html/2610.10457) | §2.1 / §3.2 | [x10ngyx/MORCA](https://github.com/x10ngyx/MORCA)：公开项目仓库；尚未发现实现文件 |

## 已收录项目的状态修正

- **EchoCache**：原记录为 2026-08-08 尚未公开；本次 GitHub 公共元数据和文件树确认已公开，包含实现文件，已修正 README。
- **RACER / EVO / HeadCast**：复查仍公开且含实现文件。其他历史条目的代码状态仍保留各自原核验日期，本次未逐篇重新审计。
- **GP-Refiner**：作者声明代码 available，但本次公开仓库为空（GitHub tree 返回 409）；README 明确标为空仓库。
- **BAG / MORCA / ManifoldCache**：项目仓库可访问，但本次默认分支文件树与项目内容未发现实现发布；不标“已开源实现”。
- **WAMachine / ParaAnya**：论文声明的地址公开访问返回 404；仅保留声明地址，不推断是私有、删除或尚未创建。
- **AutoTarget**：有核心实现文件，公开仓库说明 submitted to ICLR 2027；提交不等于接收，README 用 2026 preprint 标签。

## 边界与暂不纳入主表的相邻工作

| 工作 | arXiv | 本轮处理理由 |
|---|---|---|
| AnchorCache / Beyond Attention Masks | [2608.21229](https://arxiv.org/abs/2608.21229) | 有 exact reference KV reuse，但依赖 token-layout / mask 架构转换与两阶段蒸馏恢复；超出本轮以冻结生成模型与轻量 cache 校准为主的新增范围，可作为后续架构缓存专题。 |
| ReCaVSR | [2609.37831](https://arxiv.org/abs/2609.37831) | 新训练 one-step VSR、learned cache routing、discriminator 与 decoder 联合栈；2.72× 无法解释为 cache 单项收益。 |
| UnStep | [2609.32518](https://arxiv.org/abs/2609.32518) | 主体是少步运行、attention window 与多项运行时优化；复用 clean-cache pass 是质量修复的一部分，本轮不作为独立 feature-cache 方法纳入。 |
| Sparse-WAM | [2609.38984](https://arxiv.org/abs/2609.38984) | 主体为 action-guided token pruning，Pilot 跨步复用 token selection；暂列相邻工作，避免把所有 pruning 都扩成缓存主线。 |
| WaveAlign | [2609.34814](https://arxiv.org/abs/2609.34814) | “cache-aware”指 GPU L2 locality 与 query-row 排序，不是扩散特征状态复用；属于纯 attention kernel / scheduling 优化。 |
| AViTS | [2608.17995](https://arxiv.org/abs/2608.17995) | 动态分辨率与 selective upsampling；作者明确称与 feature caching 正交。 |
| LeanGRPO | [2609.03528](https://arxiv.org/abs/2609.03528) | 复用训练计算图 / 梯度，目标是 diffusion RL 训练开销，超出推理缓存范围。 |
| Flash-dLLM / SpecFold / MaskAhead 等 | [2609.26796](https://arxiv.org/abs/2609.26796) · [2610.04875](https://arxiv.org/abs/2610.04875) · [2610.06996](https://arxiv.org/abs/2610.06996) | 扩散语言模型的 KV 与文本解码缓存；当前仓库主轴是图像 / 视频 / 音频 / flow 生成，暂不扩出语言模型专题。 |

## 数字与命名说明

- 同名 **SpectralCache** 必须按 arXiv 区分：已有 [2603.05315](https://arxiv.org/abs/2603.05315) 是 TADS / CEB / FDC；新增 [2610.02660](https://arxiv.org/abs/2610.02660) 是世界模型 SVD 子空间与奇异值预测，不合并代码或性能。
- GP-Refiner 的 19.3% 是相对 TaylorSeer 的 compute reduction；Unexpired Plan 是 per-call compute reduction；均不转换成 wall-clock speedup。
- FlashForward / ParaAnya 是多 GPU 系统口径；SCR 的 2.8–3.0× 含 graph 系统收益；Carnator 的 2.17× 限定 cache-hit 请求。
- ManifoldCache 的理论安全性依赖其约束流形假设；Koopman Observers 的实验是 CIFAR-10 / ImageNet 子集，不外推到大规模视频 DiT。
- README 同步更新全景表、提交时间线、缓存粒度、调度详述、交叉矩阵与相关评测 / 工程入口。
