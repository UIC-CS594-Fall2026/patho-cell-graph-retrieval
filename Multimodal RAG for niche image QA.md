# Multimodal RAG for niche image QA

Nov 17, 2026 · @Shihab

## Contents

1. [Summary: recommended architecture and strongest ideas](#madkzqskytv.49)
   - [The recommended pipeline, walked through one question](#madkzqskytv.2424)
   - [Seven ideas with the strongest evidence](#madkzqskytv.4993)
2. [Retrieval paradigms and pipeline designs](#madkzqskytv.7273)
   - [Five ways to turn an image question into a search](#madkzqskytv.7507)
   - [Cascades: cheap recall, expensive rerank, strict filtering](#madkzqskytv.12040)
   - [Give the generator few, curated items](#madkzqskytv.14031)
   - [Knowledge graphs help attribute lookups, not yet multi-hop questions](#madkzqskytv.15276)
   - [Training the retriever and the generator](#madkzqskytv.16171)
   - [A KB-VQA scoreboard, with the caveats attached](#madkzqskytv.17318)
3. [Lessons from niche-domain systems, failure modes, and evaluation](#madkzqskytv.18625)
   - [Two system families recur across specialized domains](#madkzqskytv.18916)
   - [Failure mode one: the model copies the neighbors instead of looking](#madkzqskytv.22293)
   - [Failure mode two: the evidence is missing, or present but unused](#madkzqskytv.24220)
   - [Failure mode three: position, domain shift, resolution, and per-class harm](#madkzqskytv.25262)
   - [An evaluation protocol built around controlled conditions](#madkzqskytv.26447)
   - [Cold start: small curated galleries before fine-tuning](#madkzqskytv.28978)
4. [DINOv3 as the visual retrieval backbone](#madkzqskytv.31397)
   - [What DINOv3 is, without the notation](#madkzqskytv.31661)
   - [Same-size comparisons show a 40-point lead on instance retrieval](#madkzqskytv.34553)
   - [Text queries need a text-aligned partner encoder](#madkzqskytv.36474)
   - [Patch features make part- and region-level retrieval possible](#madkzqskytv.37872)
   - [Adapting DINOv3 to a domain: a four-rung ladder](#madkzqskytv.39675)
   - [Known weaknesses and the missing experiment](#madkzqskytv.41378)
5. [Where diffusion models fit](#madkzqskytv.42022)
   - [Ranking the roles by evidence](#madkzqskytv.42279)
   - [Synthetic data must beat a retrieval baseline first](#madkzqskytv.43399)
   - [Diffusion features: strong for parts, unproven for global search](#madkzqskytv.45410)
   - [Imagine-then-retrieve works for text queries if gated](#madkzqskytv.46787)
   - [Illustrations and counterfactuals, never restored evidence](#madkzqskytv.48685)
   - [Unified understand-and-generate models trade understanding for convenience](#madkzqskytv.50287)
   - [Cost and evaluation checklist](#madkzqskytv.51064)
6. [Adaptive and agentic retrieval control](#madkzqskytv.51939)
   - [Text RAG split the retrieval decision into four questions](#madkzqskytv.52208)
   - [Images make the "don't retrieve" branch matter more](#madkzqskytv.54471)
   - [Three control architectures with very different costs](#madkzqskytv.56189)
   - [Zoom and crop are retrieval-like actions](#madkzqskytv.58365)
   - [Routers are accurate in-domain and brittle outside it](#madkzqskytv.59383)
   - [A concrete action space for niche image QA](#madkzqskytv.60151)
7. [Vector indexes, quantization, and storage design](#madkzqskytv.61183)
   - [Below a million images, exact search is a serious option](#madkzqskytv.61428)
   - [Quantize, oversample, rescore](#madkzqskytv.63542)
   - [Patch multi-vectors must be merged, compressed, and used only for reranking](#madkzqskytv.65166)
   - [Regions: separate points with parent pointers, or entity-level vector lists](#madkzqskytv.67581)
   - [Selective filters break graph indexes unless planned for](#madkzqskytv.68554)
   - [Hybrid fusion and named vectors](#madkzqskytv.69810)
   - [Choosing a database by workload, not leaderboard](#madkzqskytv.70765)
8. [Research ideas and open problems](#madkzqskytv.72482)
9. [Sources](#madkzqskytv.79403)
   - [Architectures, embedders, rerankers and KB-VQA](#madkzqskytv.79569)
   - [Domain case studies, failure modes and evaluation](#madkzqskytv.80201)
   - [DINOv3 and self-supervised encoders](#madkzqskytv.80636)
   - [Diffusion models](#madkzqskytv.81053)
   - [Adaptive and agentic retrieval](#madkzqskytv.81849)
   - [Vector indexes, quantization and databases](#madkzqskytv.82249)

## Summary: recommended architecture and strongest ideas

The best-supported design for RAG in a niche image domain is to **retrieve labeled visual exemplars together with their expert text, then have a vision-language model (VLM) check the query image against a small, filtered shortlist**. Captioning the image and searching text is the weaker approach. On fine-grained identification, this pattern lifted 7B open VLMs by roughly 15 to 35 points. Bird species accuracy went from **26.8% to 42.2%** ([VR-RAG](https://arxiv.org/html/2505.05635v2)), and a 10,000-species iNaturalist task went from **24.87% to 60.07%** ([DeepTaxon](https://arxiv.org/html/2604.24029)).

The same studies show how badly the naive version fails. Plain CLIP retrieval pushed the bird VLM *below* its no-retrieval score, to 21.7% ([VR-RAG](https://arxiv.org/html/2505.05635v2)). Retrieved, not ground-truth, images made most open VLMs worse on MRAG-Bench ([MRAG-Bench](https://arxiv.org/abs/2410.08182)). In medical RAG, about half of all errors come from the model leaning on retrieved reports instead of the image ([RULE](https://arxiv.org/html/2407.05131v2)).

So a good retriever is only one of four required parts. The other three are:

- **A relevance filter.** Removing it cost 11.9 points on Encyclopedic-VQA ([ReflectiVA](https://arxiv.org/abs/2411.16863)).
- **A retrieval gate.** It protects perception questions, where always retrieving dropped one VLM from 50.13 to 30.37 ([knowledge boundary](https://arxiv.org/html/2502.18023v3)).
- **An evaluation protocol that detects copying.**

DINOv3 is the strongest encoder available for the image-to-image half of retrieval. At ViT-L size it scores **63.1 on Oxford-Hard against 21.4 for SigLIP 2** ([DINOv3](https://arxiv.org/html/2508.10104v1)). It is weaker on text-to-image search, so the index should hold both kinds of embedding. Diffusion models belong offline: hard negatives, rare-class data, and explanations. Real retrieved images beat synthetic ones by a wide margin, and online generation costs 12–27 seconds per round. At 10^3 to 10^7 images, the vector database is rarely the bottleneck. The real costs are patch-level multi-vectors, selective metadata filters, and re-embedding when you change models.

*How to read this doc.* The findings come from research notes current to November 2026, and every sourced finding has an inline link. Text starting with **Inference:** is synthesis, design extrapolation, or rough arithmetic that no paper reported. **Comparability:** marks numbers from different knowledge bases, judges, splits, or protocols that should not be ranked against each other. Several headline results come from unreplicated preprints and are called self-reported.

### The recommended pipeline, walked through one question

Imagine an agricultural extension service. A farmer uploads a field photo of a beetle on a leaf and asks, "What is this, and should I spray?" The table follows that question through each stage.

&#91;embedded content: recommended pipeline · gate, crop, recall, rerank, filter, verify\]

Perception questions exit at the gate; everything else is recalled broadly, cut to a short filtered list, and verified against the image before answering.

| Stage | What happens to the beetle question | Default component | Evidence status |
| --- | --- | --- | --- |
| Knowledge base (offline) | A gallery holds 1–5 expert-verified photos per pest species, plus a short visual-trait summary, management text, and fixed metadata (taxonomy, region, life stage) | Labeled exemplar gallery, trait text, metadata schema | Gallery pattern sourced; schema design is inference |
| Gate | The VLM's sampled answers disagree on the species, so the system retrieves. "How many beetles are there?" would skip retrieval | Sampling consistency or a knowledge-boundary LoRA | Sourced |
| Subject crop | The beetle is cropped so the leaf does not dominate the embedding | Grounding tool or subject-extraction pass | Sourced as a component |
| Recall | The crop searches the gallery image-to-image, the full image searches trait text, and captions are searched by BM25; the ranked lists are fused | DINOv3 global vector, SigLIP 2 or PE vector, BM25, reciprocal-rank fusion, top \~100 | Components sourced; the combination is inference |
| Rerank | 100 candidates shrink to 10 species with up to 3 exemplars each | DINOv3 patch matching and/or a VLM yes/no reranker | Reranker gains sourced; DINOv3 patch rerank is inference |
| Filter | Candidates with irrelevant evidence are dropped, and the list is cut at the first big drop in similarity | Critic model or relevance tokens; gap-based truncation | Sourced |
| Generate | The VLM compares the beetle with the exemplars, names a species or answers "none match", then answers "should I spray?" from that species' management text | Open 7B VLM tuned against copying | Sourced |
| Escalate | If the answer needs local regulations, an agent runs a text search and then stops | Tool-using agent with a stopping rule | Sourced |

Sources for each stage:

- **Gallery pattern:** DeepTaxon reaches 44.45% with a single exemplar per species ([DeepTaxon](https://arxiv.org/html/2604.24029)). RAR uses three unlabeled images per category ([RAR](https://liuziyu77.github.io/RAR)).
- **Gate:** sampling-based knowledge boundaries cut retrieval to about half of queries while beating always-retrieve ([knowledge boundary](https://arxiv.org/html/2502.18023v3)).
- **Subject crop:** ReAG adds a retrieval pass on the question's subject ([ReAG](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Compagnoni_ReAG_Reasoning-Augmented_Generation_CVPR_2026_supplemental.pdf)). DeepMMSearch-R1 crops with Grounding-DINO before image search ([DeepMMSearch-R1](https://arxiv.org/html/2510.12801v1)).
- **Gap-based truncation:** from MMed-RAG ([MMed-RAG](https://www.arxiv.org/pdf/2410.13085)).
- **Stopping rule:** PMSR stops when successive queries become near-duplicates ([PMSR](https://aclanthology.org/2026.acl-long.881.pdf)).

### Seven ideas with the strongest evidence

**Exemplar galleries plus VLM verification beat scale.** Without retrieval, Qwen2.5-VL-72B scores 3.53% on Cars and 4.23% on Aircraft. A retrieval-augmented 7B model trained to compare the query against retrieved exemplars moves Flowers from 40.59% to 99.17% ([DeepTaxon](https://arxiv.org/html/2604.24029)). Generic VLMs know too little about niche classes, and a gallery supplies that knowledge without retraining.

**Filter what you retrieve, and gate when you retrieve.** On knowledge-only benchmarks, the timing decision barely matters. Forcing ReflectiVA to always retrieve changed E-VQA from 35.5 to 35.3, while removing its relevance tokens dropped it to 23.6 ([ReflectiVA](https://arxiv.org/abs/2411.16863)). On mixed workloads the picture flips: always retrieving collapsed captioning-style accuracy on NoCaps from 50.13 to 30.37 ([knowledge boundary](https://arxiv.org/html/2502.18023v3)). A real product sees both kinds of question, so it needs both mechanisms.&#32;

**First-stage recall caps everything downstream.** On E-VQA, the best CLIP-class encoder finds the right page in its top 20 only 49.8% of the time ([ReflectiVA](https://arxiv.org/abs/2411.16863)). DeepTaxon's accuracy is bounded by Pass@16 retrieval at about 79.8% ([DeepTaxon](https://arxiv.org/html/2604.24029)). On MRAG-Bench, downstream accuracy tracks Recall@5 with a reported \~95% correlation ([MRAG-Bench](https://arxiv.org/abs/2410.08182)).

**Copy tests are both diagnostics and free training data.** MMed-RAG replaced the query image with unrelated noise while keeping the retrieved reports. **55.08% of answers stayed "correct"**, which means the model was answering from the text. After preference training on pairs mined from these failures, that figure fell to 28.19% ([MMed-RAG](https://www.arxiv.org/pdf/2410.13085)).

**Classification accuracy is the wrong yardstick for picking a retrieval encoder.** DINOv3, SigLIP 2 and PE are within about a point of each other on ImageNet linear probing. On instance retrieval, DINOv3 leads them by roughly 40 points at the same model size ([DINOv3](https://arxiv.org/html/2508.10104v1)).

**Retrieve real images before generating fake ones.** On FGVC-Aircraft, 139K retrieved LAION images added 17.8 linear-probe points, while 139K Stable Diffusion images added 3.8 ([Geng et al.](https://arxiv.org/html/2406.05184v2)).

**Store one global vector per image and rerank with compressed patches.** Merging patch embeddings keeps 98.2% of retrieval quality at 11.8% of the memory ([Light-ColPali](https://arxiv.org/html/2506.04997v1)). Filters need planning: on a plain HNSW graph with a 1% filter, recall fell to 0.1% ([Qdrant](https://qdrant.tech/articles/filtered-vector-search-acorn/)).

## Retrieval paradigms and pipeline designs

Five retrieval paradigms coexist, and the right one depends on what the corpus looks like. Around them, a three-to-four-stage cascade (recall, rerank, filter, generate) is the dominant design.

### Five ways to turn an image question into a search

**Text conversion.** The simplest approach captions or OCRs every image and runs ordinary text search. On text-dense pages it loses heavily. On ViDoRe, OCR plus BM25 scores 65.5 nDCG@5 and captioning plus BM25 scores 65.1, against **81.3 for ColPali**, and captioning "can take dozens of seconds per page" at index time ([ColPali](https://arxiv.org/abs/2407.01449)). Text pipelines are not obsolete, though. On ViDoRe V3, adding a strong text reranker (zerank-2) gave a textual pipeline **+13.2 nDCG@10**, while the best visual retrievers beat the best textual one by 59.8 to 51.0 ([ViDoRe V3](https://arxiv.org/abs/2601.08620)). A variant first names the entity in the image and then looks up its article. Google Lens plus PaLM reaches 48.0 on E-VQA, against 19.7 for PaLM alone ([EchoSight](https://arxiv.org/abs/2407.12735)).

**A shared image-text embedding space.** CLIP-family dual encoders place images and text in one vector space, so you can search in either direction. Which direction wins depends on the corpus, which is the most practically useful finding here:

| Benchmark (KB size) | Image-to-image R@1 | Image-to-text R@1 | Better direction |
| --- | --- | --- | --- |
| E-VQA (2M pages) | 15.6 | 10.1 | Image-to-image |
| InfoSeek (100k pages) | 29.6 | 56.1 | Image-to-text |

These numbers use EVA-CLIP-8B; the text side is the article title plus summary ([ReflectiVA](https://arxiv.org/abs/2411.16863)).

A second issue is the "modality gap": matched image-text pairs sit far apart in CLIP space. The mean cosine similarity of matched Flickr30K pairs is 0.15 for OpenAI CLIP and 0.71 for jina-embeddings-v4 ([jina-v4](https://arxiv.org/abs/2506.18902)). On natural-image corpora CLIP still holds up: it had the best Recall@5 on MRAG-Bench (60.46), ahead of newer VLM embedders ([MRAG-Bench](https://arxiv.org/abs/2410.08182)).

**VLM-based embedders.** These fine-tune a whole VLM to output embeddings, in two forms.

*Single-vector* models pool the VLM's states into one vector per image. Examples are VisRAG-Ret, DSE, VLM2Vec, GME, and Qwen3-VL-Embedding.

*Multi-vector "late interaction"* models such as ColPali keep one small vector per image patch. They score a page by summing, over the query's tokens, each token's best match on the page. Worked example: say the query has three tokens, "red", "spotted" and "wing", and the page has about 1,000 patch vectors. For each token, find the single best-matching patch: 0.9 for "red", 0.7 for "spotted", 0.8 for "wing". The page scores 2.4. A page with a similar overall look but no spotted region scores lower, because no patch matches "spotted" well. This "MaxSim" operator is why late interaction is good at small details. Its cost is storage: ColPali keeps about **256 KB per page**, versus **4.5 KB** for VisRAG-Ret's single 2,304-dimension vector ([VisRAG](https://arxiv.org/abs/2410.10594)).

The gap between the two forms has narrowed:

- On the visual-document average, single-vector Qwen3-VL-Embedding-8B scores 75.8 against 77.7 for the best late-interaction model ([Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720)).
- Within one model, jina-v4 scores 84.11 dense and 90.17 with late interaction on ViDoRe ([jina-v4](https://arxiv.org/abs/2506.18902)).
- One pitfall: an MLLM bi-encoder can underperform a smaller CLIP on cross-modal retrieval ("modality bias"). MM-Embed fixes this with modality-aware hard negatives ([MM-Embed](https://arxiv.org/abs/2411.02571)).

**Feeding raw retrieved images to the generator.** Instead of converting retrieved pages to text, pass the images themselves. VisRAG reports about **40% relative gain** over text-based RAG with MiniCPM-V 2.6, and InfoVQA rises from 25% to 51% ([VisRAG](https://arxiv.org/abs/2410.10594)). On ViDoRe V3's hard queries, image context beats text context by 2.4 points with oracle pages and 2.8 with retrieved ones ([ViDoRe V3](https://arxiv.org/abs/2601.08620)). On natural images, image RAG also beats text RAG on MRAG-Bench ([MRAG-Bench](https://arxiv.org/abs/2410.08182)).

**Dedicated knowledge-based VQA (KB-VQA) systems.** These stack rerankers, filters and reasoning on top of CLIP-class retrieval over Wikipedia-scale knowledge bases:

- RA-VQA trains the retriever jointly with the answer generator ([RA-VQA](https://arxiv.org/abs/2210.03809)).
- PreFLMR uses late interaction. It lifts E-VQA PR@5 to 73.7, against 10.4 for zero-shot CLIP, and downstream accuracy with a BLIP-2 generator from 19.80 to 54.45 ([PreFLMR](https://arxiv.org/abs/2402.08327)).
- Wiki-LLaVA is hierarchical: it retrieves the entity page, then passages inside it ([Wiki-LLaVA](https://arxiv.org/abs/2404.15406)).
- EchoSight, ReflectiVA, mR²AG, OMGM and ReAG appear in the cascade discussion below.

**Inference:** For natural-image niche domains such as specimens, parts, crops and defects, run image-to-image and image-to-text search in parallel and fuse the results. The E-VQA/InfoSeek split shows that neither direction wins everywhere, and you rarely know in advance which one your corpus favors. For document-like imagery (manuals, labels, drawings, charts), use a VLM page embedder and give the generator the page images.

### Cascades: cheap recall, expensive rerank, strict filtering

A cheap bi-encoder recalls 20–100 candidates, a multimodal reranker narrows them, and a relevance filter or passage localizer picks what the generator sees.

Reranking is the biggest single lever on KB-VQA. EchoSight's reranker is a BLIP-2 Q-Former trained with 24 hard negatives per query. It lifts E-VQA R@1 from **13.3 to 36.5**, and hard negatives alone account for 31.4 to 36.5 of that ([EchoSight](https://arxiv.org/abs/2407.12735)). Reranking more candidates helps less and less while throughput collapses. R@1 is 29.4 when reranking the top 5, 36.5 at top 20, 38.8 at top 100 and 39.8 at top 500, while throughput falls from about 1.66 to 0.05 queries per second ([EchoSight](https://arxiv.org/abs/2407.12735)).

Reranker quality varies enormously:

- The VLM cross-encoder Qwen3-VL-Reranker lifted ViDoRe v3 from 52.9 to 60.8 at 2B and to 66.7 at 8B ([Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720)).
- The older jina-reranker-m0 added only +0.2 nDCG@10 on the same benchmark family ([ViDoRe V3](https://arxiv.org/abs/2601.08620)).

Relevance filtering inside or before the generator is the next lever:

- Removing ReflectiVA's relevance tokens cuts E-VQA from 35.5 to 23.6 and InfoSeek from 40.1 to 31.4 ([ReflectiVA](https://arxiv.org/abs/2411.16863)).
- ReAG's 3B critic model reaches 94.6% recall and 95.7% specificity for passage relevance, and cuts the average passage count from 128.6 to 5.7 ([ReAG](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Compagnoni_ReAG_Reasoning-Augmented_Generation_CVPR_2026_supplemental.pdf)).
- mR²AG multiplies retrieval similarity, relevance probability and answer confidence into one score. That reaches 40.2 on InfoSeek, against 36.7 for answer confidence alone ([mR²AG](https://arxiv.org/html/2411.15041v1)).

**Inference:** A reranker only reorders what recall found. EchoSight's R@20 is 48.8 after reranking, against 49.8 R@20 for raw EVA-CLIP-8B in ReflectiVA's table. The setups differ, but the point holds: the reranker cannot rescue the half of queries whose evidence never entered the shortlist. Once a decent reranker is in place, better first-stage recall (fused search directions, subject crops) probably pays more than a better reranker. No fetched paper tests whether reranker gains and filter gains add up when stacked.

### Give the generator few, curated items

Long-context VLMs make it tempting to send dozens of images, and the evidence says not to:

- On MRAG-Bench, accuracy rises up to about 10 images, and 20 images score 0.29 points worse than 10 ([MRAG-Bench](https://arxiv.org/abs/2410.08182)).
- Visual Haystacks finds that VLMs struggle with unrelated images, reason poorly across images, and are biased by where the key image sits in the context ([Visual Haystacks](https://arxiv.org/abs/2407.13766)).
- The best k depends on the dataset: ReflectiVA's best k is 10 on E-VQA and 1 on InfoSeek ([ReflectiVA](https://arxiv.org/abs/2411.16863)).
- Open and proprietary models react very differently to noisy retrieval. With retrieved images, GPT-4o gained only +0.28 on MRAG-Bench while most open models lost accuracy; with ground-truth images GPT-4o gained +5.82 ([MRAG-Bench](https://arxiv.org/abs/2410.08182)).
- Attribution is weak even for frontier models. On ViDoRe V3, the best bounding-box grounding F1 is **0.089**, against **0.602** agreement between human annotators ([ViDoRe V3](https://arxiv.org/abs/2601.08620)).

**Inference:** For entity and attribute lookups, retrieved knowledge can go to the generator as text, with images used only to retrieve; EchoSight does this with a text-only generator. For diagrams, labels and document-like images, pass the images. Either way, k should be small and calibrated per question type, roughly 1 to 10.

### Knowledge graphs help attribute lookups, not yet multi-hop questions

mKG-RAG builds a multimodal knowledge graph for each document offline:

- An MLLM extracts entities and relations from the text.
- A scene-graph model extracts objects and relations from the images.
- A matching step attaches image regions to text entities.

At query time it retrieves documents, then entities and relations with one-hop expansion. Graph retrieval beat chunk retrieval by **8.1 points on E-VQA and 7.5 on InfoSeek**. Two-hop gains were roughly half the single-hop gains, and compute was 5.12× a zero-shot baseline, against 4.49× for naive RAG ([mKG-RAG](https://arxiv.org/abs/2508.05318)).

**Inference:** Niche domains often already have curated metadata, such as species taxonomies, part catalogs or defect codes. That metadata can stand in for the LLM-extracted graph, giving structure-aware retrieval without the extraction cost. Nobody has tested this.

### Training the retriever and the generator

Retriever training is almost always contrastive (InfoNCE) with in-batch negatives plus mined hard negatives. Newer pipelines also distill the reranker's scores into the embedder: Qwen3-VL-Embedding trains its embedder on the reranker's score distribution and then merges checkpoints ([Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720)). PreFLMR's scaling lessons transfer well ([PreFLMR](https://arxiv.org/abs/2402.08327)):

- Scaling the vision encoder from ViT-B to ViT-G helps but plateaus.
- A BERT-Base text encoder is enough; larger ones overfit.
- A bigger mapping network does not help.

Generator training for robustness to retrieval has three ingredients:

- **Mixed-quality context.** ReflectiVA pairs a positive passage with a hard negative from the same page and a soft negative from another page.
- **Noise injection.** RoRA-VLM and RagVL inject noise during training ([RoRA-VLM](https://arxiv.org/abs/2410.08876); [RagVL](https://arxiv.org/abs/2407.21439)).
- **Reinforcement learning over reasoning traces.** ReAG-7B scores 55.5 on E-VQA single-hop against 48.3 for ReflectiVA-7B under the same Google Lens retrieval. That gap bundles a better critic, subject-level retrieval, and SFT plus GRPO, so the RL contribution alone is not isolated ([ReAG](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Compagnoni_ReAG_Reasoning-Augmented_Generation_CVPR_2026_supplemental.pdf)).

### A KB-VQA scoreboard, with the caveats attached

| System | First-stage retrieval | E-VQA single-hop | InfoSeek (all) |
| --- | --- | --- | --- |
| LLaVA-1.5-7B, no retrieval | none | 16.3 | 9.5 |
| GPT-4V, no retrieval | none | 26.9 | 14.6 |
| Wiki-LLaVA | CLIP ViT-L + Contriever | 17.7 | 28.9 |
| EchoSight (reproduced, visual) | EVA-CLIP-8B + Q-Former rerank | 26.4 | 18.8 |
| ReflectiVA | EVA-CLIP-8B | 35.5 (visual) | 40.1 (text) |
| mKG-RAG (fine-tuned) | QM-Retriever + graph | 38.4 | 40.5 |
| mR²AG | Google Lens | 55.9 | — |
| ReAG-7B | Google Lens | 55.5 | — |

Sources: [ReflectiVA](https://arxiv.org/abs/2411.16863), [mKG-RAG](https://arxiv.org/abs/2508.05318), [ReAG supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Compagnoni_ReAG_Reasoning-Augmented_Generation_CVPR_2026_supplemental.pdf).

**Comparability:** treat this table as orientation, not a ranking.

- **Wiki-LLaVA** is reported as 17.7, 21.8 or 26.4 on E-VQA depending on which paper reproduces it.
- **EchoSight** reports 31.3 on InfoSeek for itself, while ReflectiVA's reproduction gets 30.4 (text) or 18.8 (visual).
- **InfoSeek knowledge bases differ:** ReflectiVA uses a 100k-page subset, while EchoSight filtered its own because the original was not released ([EchoSight](https://arxiv.org/abs/2407.12735)).
- **Google Lens rows are not comparable to CLIP rows.** Web-scale entity search is a much stronger retriever than any open encoder over a Wikipedia knowledge base.

The one anchor everyone shares is E-VQA's own result: PaLI scores 13.0% and oracle retrieval 87.0% on single-hop questions ([Encyclopedic-VQA](https://arxiv.org/abs/2306.09224)). Retrieval is still the gap.

## Lessons from niche-domain systems, failure modes, and evaluation

Specialized systems that work retrieve either expert text or labeled exemplar images. Their main failure is the VLM copying retrieved labels instead of looking at the image, which only controlled evaluation conditions expose.

### Two system families recur across specialized domains

- **Image retrieves expert text.** Reports, textbook pages or encyclopedic summaries are retrieved, and the VLM conditions on them. This is common in radiology, pathology, remote sensing and art.
- **Image retrieves labeled exemplar images.** The VLM or a scoring head compares the query against them. This is effectively nearest-neighbor classification with a VLM judge, and it dominates species ID, plant disease, industrial anomaly detection and fine-grained recognition.

Domains with very high-resolution or dense imagery (whole pathology slides, ultra-high-resolution satellite scenes, engineering drawings) add a crop or region-selection step first.

| System | Domain | Retriever | Knowledge source | Headline gain |
| --- | --- | --- | --- | --- |
| [RULE](https://arxiv.org/html/2407.05131v2) | Radiology, ophthalmology | ResNet-50 + BioClinicalBERT, contrastive fine-tune | Training-split reports | IU-Xray VQA 75.47 → 87.84 |
| [MMed-RAG](https://www.arxiv.org/pdf/2410.13085) | Radiology, ophthalmology, pathology | BiomedCLIP domain router + per-domain retrievers | Training-split reports | VQA +18.5% and report generation +69.1% average (relative) |
| [Path-RAG](https://arxiv.org/pdf/2411.17073) | Pathology VQA | Nuclei-count patch ranking, no external KB | Image patches | PathVQA-Open H&E 36.4 → 64.1 |
| [Patho-AgenticRAG](https://www.alphaxiv.org/overview/2508.02258) | Pathology | ColQwen2 pages in Milvus HNSW | \~200k textbook page images | Quilt-VQA 60.93 → 75.80 (self-reported) |
| [RS-RAG](https://arxiv.org/html/2504.04988v2) | Remote-sensing landmarks | CLIP image + text, Qdrant, fused score | Imagery + 39 sensor variables + Wikipedia | Classification 0.61 → 0.79 |
| [Snap'n Diagnose](https://arxiv.org/html/2408.14723v1) | Plant disease | CLIP + fine-tuned MLP | 18k in-the-wild images, 89 classes | Top-1 40.92 → 67.32 |
| [VR-RAG](https://arxiv.org/html/2505.05635v2) | Bird species | CLIP/OpenCLIP/SigLIP ensemble, DINOv2 rerank | Wikipedia trait summaries + ≤3 exemplars | 26.8 → 42.2 average |
| [DeepTaxon](https://arxiv.org/html/2604.24029) | 10k species | OpenCLIP ViT-H + FAISS | \~400k labeled exemplars | 24.87 → 60.07 (self-reported) |
| [ReMP-AD](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.pdf) | Industrial defects | Frozen CLIP patch memory | K normal images | VisA 4-shot I-AUC 94.1 vs 89.1 |
| [Blueprint](https://arxiv.org/html/2602.13345) | Engineering drawings | YOLOv8 title-block regions + VLM OCR + BM25/dense fusion | \~770k legacy files | nDCG@3 0.626 vs 0.521 (self-reported) |
| [RAG-Driver](https://arxiv.org/html/2402.10828v2) | Driving | Learned video + control embedding | BDD-X clips + explanations | Out-of-domain action CIDEr 5.7 → 48.9 |

**Comparability:** the "headline gain" column mixes accuracy, relative improvements, CIDEr, AUC and nDCG. These numbers say "retrieval helped a lot here"; they do not rank the systems. VR-RAG's arXiv ID also appears in search results under a different title ("Neural Catalog..."), so check the version before citing it by name.

Three design lessons recur across the table:

- **Medicine trains small in-domain retrievers.** The medical systems train compact dual encoders on in-domain image-report pairs rather than relying on an off-the-shelf biomedical CLIP. BiomedCLIP appears only as a coarse domain router ([MMed-RAG](https://www.arxiv.org/pdf/2410.13085)).
- **Train the retriever on task similarity, not visual similarity.** FactMM-RAG mines fact-similar report pairs with RadGraph to train its retriever, gaining up to 6.5% F1-CheXbert ([FactMM-RAG](https://aclanthology.org/2025.naacl-long.28)). RAG-Driver picks training triplets by text similarity of the expert explanations. It found that applying in-context examples only at inference, without training on that format, "produced unusable outputs" ([RAG-Driver](https://arxiv.org/html/2402.10828v2)).
- **Zooming matters more than clever region selection.** In Path-RAG, random patches already gave 46.2, against 47.4 for knowledge-guided patches ([Path-RAG](https://arxiv.org/pdf/2411.17073)).

### Failure mode one: the model copies the neighbors instead of looking

The best-documented failure is over-reliance: the VLM reads the retrieved text and largely ignores the query image.

- **RULE** measured over-reliance ratios of 47.42% (IU-Xray), 47.44% (FairVLMed) and **58.69%** (MIMIC-CXR) of RAG errors ([RULE](https://arxiv.org/html/2407.05131v2)).
- **MMed-RAG's copy-reference test** swaps the query image for unrelated noise and keeps the retrieved reports. 55.08% of answers stayed correct, and 43.31% of answers that were right without RAG became wrong with it. Attention analysis showed RAG shifting attention toward the retrieved answer even when that answer was wrong ([MMed-RAG](https://www.arxiv.org/pdf/2410.13085)).
- **General multimodal in-context learning is text-driven** (Baldassini et al.). Randomizing the demo *images* barely mattered, while randomizing the demo *answers* hurt a lot, and the model copied the last demo's answer in about 12% of VQA cases. The authors caution this may not hold for stronger models ([Baldassini et al.](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Baldassini_What_Makes_Multimodal_In-Context_Learning_Work_CVPRW_2024_paper.pdf)).

The fix that worked is preference training on the failures themselves:

- RULE keeps cases the model answered correctly without retrieval but wrongly with it (1,106 to 2,259 pairs per dataset) and trains with DPO ([RULE](https://arxiv.org/html/2407.05131v2)).
- MMed-RAG adds pairs whose rejected answer came from a noised image plus the same reports. After training, copy-reference fell to 28.19% and over-reliance to 8.38% ([MMed-RAG](https://www.arxiv.org/pdf/2410.13085)).

Worked example of the over-reliance ratio: take 200 validation questions the VLM answers correctly with no retrieval. Turn retrieval on and 30 of them flip to wrong, so the ratio is 15%. Those 30 (question, correct answer, wrong answer) triples are ready-made DPO training pairs.

**Inference:** in a niche domain where retrieved exemplars carry class labels, expect the VLM to behave largely as a label-copier or majority voter over its neighbors. System accuracy then roughly equals retrieval precision at small k, unless the generator is trained to verify against the image.

### Failure mode two: the evidence is missing, or present but unused

**Retrieval misses:**

- On Visual-RAG, there is about a 17–18% chance that no clue image appears in the top 30 ([Visual-RAG](https://alphaxiv.org/overview/2502.16636v2)).
- DeepTaxon's zero-shot VLMs "get distracted by candidates", reason from habitat cues that are not visible in the image, and reject correct matches ([DeepTaxon](https://arxiv.org/html/2604.24029)).

**Fine-grained confusion:**

- VR-RAG reaches 68.0% at genus level but only 42.2% at species level, so most errors land in the right genus ([VR-RAG](https://arxiv.org/html/2505.05635v2)).
- More candidates can hurt: InternVL dropped from 24.3% at k=5 to 21.7% at k=10 ([VR-RAG](https://arxiv.org/html/2505.05635v2)).

**Under-use of visual evidence:** even when the right evidence is present, models extract little from it. On MRAG-Bench, ground-truth visual knowledge improved GPT-4o by 5.82 points and humans by **33.16** ([MRAG-Bench](https://mlanthology.org/iclr/2025/hu2025iclr-mragbench)).

**Knowledge vs. perception:** AgroBench's error analysis attributes about 52% of GPT-4o's agricultural errors to missing knowledge and 33% to perception ([AgroBench](https://openaccess.thecvf.com/content/ICCV2025/papers/Shinoda_AgroBench_Vision-Language_Model_Benchmark_in_Agriculture_ICCV_2025_paper.pdf)). Retrieval addresses the first share; only better visual comparison addresses the second.

### Failure mode three: position, domain shift, resolution, and per-class harm

- **Position bias.** Accuracy over evidence position follows a U-shaped curve, so the middle of the list is under-used. Images make the bias stronger than text alone ([Who is in the Spotlight](https://www.emergentmind.com/papers/2506.11063)).
- **Domain shift.** Plant-disease models trained on lab images "decline sharply" on in-the-wild photos ([Snap'n Diagnose](https://arxiv.org/html/2408.14723v1)).
- **Resolution.** Small objects in huge images break standard pipelines. Restricting a remote-sensing benchmark to tiny targets drops SkysenseGPT detection mAP from 28.25 to 5.07 ([ImageRAG-RS](https://arxiv.org/abs/2411.07688)).
- **Gains confined to one subdomain.** Path-RAG's gains held only on H&E-stained images. On other stains its patch pipeline matched or fell below baseline, and GPT-4 aggregation produced confident speculation from partly unrelated patches ([Path-RAG](https://arxiv.org/pdf/2411.17073)).
- **OCR on dense documents.** On engineering drawings, region-restricted OCR beat full-page OCR, and document-only queries failed most often, at a 35.9% loss rate ([Blueprint](https://arxiv.org/html/2602.13345)).
- **Per-class harm.** Retrieval can hurt specific categories even when it helps on average. RS-RAG scored 0.23 against the base model's 0.52 on "Theater" ([RS-RAG](https://arxiv.org/html/2504.04988v2)). Only per-class reporting reveals this.

### An evaluation protocol built around controlled conditions

The strongest evaluation designs in this literature have three layers:

1. **Retrieval metrics** (Recall/Hit@k, nDCG, MRR), computed on whatever unit the generator actually consumes: image, page or region.
2. **Answer quality.** Accuracy for closed questions; for free text, a validated judge or domain-entity metrics such as F1-RadGraph.
3. **Controlled conditions** that isolate how evidence is used, assembled below from several papers.

| Condition | What it reveals | Source |
| --- | --- | --- |
| No retrieval | Baseline prior knowledge | [Visual-RAG](https://alphaxiv.org/overview/2502.16636v2) |
| Ground-truth evidence (oracle) | Headroom if retrieval were perfect | [MRAG-Bench](https://mlanthology.org/iclr/2025/hu2025iclr-mragbench) |
| Top-k retrieval | Realistic system behavior | [Visual-RAG](https://alphaxiv.org/overview/2502.16636v2) |
| One clue plus K−1 distractors | Robustness to noise; Visual-RAG's gCUE normalizes gains against the oracle | [Visual-RAG](https://alphaxiv.org/overview/2502.16636v2) |
| Swapped or noised query image | Copying from retrieved text | [MMed-RAG](https://www.arxiv.org/pdf/2410.13085) |
| Image removed | How far text priors alone go | [AgroBench](https://openaccess.thecvf.com/content/ICCV2025/papers/Shinoda_AgroBench_Vision-Language_Model_Benchmark_in_Agriculture_ICCV_2025_paper.pdf) |
| Pass@k of the true class | Hard ceiling on end accuracy | [DeepTaxon](https://arxiv.org/html/2604.24029) |

Judges need validation before anyone trusts them. Agreement between LLM judges and humans varied from quadratic **κ = 0.31** for Blueprint's retrieval judges ([Blueprint](https://arxiv.org/html/2602.13345)) to **F1 = 0.92** for Visual-RAG's answer judge ([Visual-RAG](https://alphaxiv.org/overview/2502.16636v2)). MLLM-as-a-Judge found that GPT-4V's scoring correlates with humans at only 0.490 Pearson, and documented position, length and egocentric biases; pairwise comparison was the most reliable mode ([MLLM-as-a-Judge](https://alphaxiv.org/overview/2402.04788v3)). RAG-Check separates two sources of hallucination, irrelevant retrievals versus misdescribed retrieved images, and trains a scorer for each ([RAG-Check](https://huggingface.co/papers/2501.03995)).

**Inference:** a minimal protocol for a niche system:

- the five core conditions above;
- an over-reliance ratio and counts of answers that retrieval helped versus hurt;
- a shuffled-order run for position sensitivity;
- hierarchical scoring (genus vs. species, defect family vs. type) to separate near-misses from gross errors, as VR-RAG does;
- per-class breakdowns.

Strictly exclude evaluation images and their near-duplicates from the index. RS-RAG, as read, does not state whether it did ([RS-RAG](https://arxiv.org/html/2504.04988v2)); the medical systems retrieve only from training splits ([MMed-RAG](https://www.arxiv.org/pdf/2410.13085)).

A useful question taxonomy, synthesized from AgroBench's seven task types, MechVQA's three levels and RS-RAG's VQA subtypes: identification; attribute reading; counting/localization; comparison against an exemplar; knowledge/management; multi-step judging. Tag each question with whether it needs the image, the knowledge base, or both ([AgroBench](https://openaccess.thecvf.com/content/ICCV2025/papers/Shinoda_AgroBench_Vision-Language_Model_Benchmark_in_Agriculture_ICCV_2025_paper.pdf); [MechVQA](https://huggingface.co/papers/2605.30794)).

### Cold start: small curated galleries before fine-tuning

The evidence favors starting with retrieval over a small expert gallery rather than fine-tuning first:

- DeepTaxon with one exemplar per species reaches 44.45% against a 24.87% zero-shot baseline, and swapping the index adapts it to new domains without retraining ([DeepTaxon](https://arxiv.org/html/2604.24029)).
- RAR's fine-grained memory uses three unlabeled images per category, lifting 4-shot top-1 from 57.0 to 63.2 ([RAR](https://liuziyu77.github.io/RAR)).
- ReMP-AD needs only one to four *normal* reference images for defect detection ([ReMP-AD](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.pdf)).
- Adding retrieved open-world data cuts both ways. In SWAT, fine-tuning on retrieved pretraining data alone underperformed zero-shot. The remedy was to fine-tune on mixed data, then retrain the classifier on the clean few-shot set ([SWAT](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Few-Shot_Recognition_via_Stage-Wise_Retrieval-Augmented_Finetuning_CVPR_2025_paper.html)).
- Knowledge text should be distilled into visually discriminative form. VR-RAG had GPT-4o condense Wikipedia articles from 552 to 76 words of visual traits before indexing ([VR-RAG](https://arxiv.org/html/2505.05635v2)).
- For document-like archives, a strict metadata schema extracted from detected regions did most of the work ([Blueprint](https://arxiv.org/html/2602.13345)).

Two methods choose how many neighbors to keep:

- **RULE** calibrates k statistically on a validation split. It keeps only the k values whose factuality risk is provably bounded, which needs a modest labeled validation set rather than training data ([RULE](https://arxiv.org/html/2407.05131v2)).
- **MMed-RAG** truncates at the first large gap between consecutive similarity scores ([MMed-RAG](https://www.arxiv.org/pdf/2410.13085)). Worked example with illustrative numbers: retrieved similarities of 0.82, 0.81, 0.79, 0.78, 0.61, 0.60. The ratios between neighbors stay near 1 until 0.78 → 0.61, a ratio of about 1.28, so the system keeps four neighbors. The authors saw this cliff around the top 9 in their data.

**Inference:** a cold-start recipe built from these pieces:

1. Collect 1–5 expert-verified exemplars per class, plus trait text in a fixed schema covering class, hierarchy, discriminative traits, acquisition conditions and provenance.
2. Index them with a text-aligned encoder for recall and DINOv3 for image-to-image reranking.
3. Have the VLM compare the query against the top classes' exemplars, with an explicit "none match" verdict (DeepTaxon's "\[Discovery\]" output is the template).
4. Calibrate k and the abstention threshold on a small validation set.
5. Mine copy-test failures as DPO pairs.
6. Grow the gallery from expert-confirmed production queries.

No paper reports numbers for that last step, active gallery curation.

## DINOv3 as the visual retrieval backbone

DINOv3 leads text-aligned encoders by about 40 points on instance retrieval at equal size, and its patch features enable region-level matching. It trails them on text-to-image search, so pair it with a text-aligned encoder.

### What DINOv3 is, without the notation

DINOv3 is Meta's self-supervised vision model, released 14 August 2025 and listed in TMLR 2026 ([DINOv3 paper](https://arxiv.org/abs/2508.10104); [ML Anthology](https://mlanthology.org/tmlr/2026/simeoni2026tmlr-dinov3/)). "Self-supervised" here means no captions and no labels. A teacher network and a student network see different crops of the same photo, and the student learns to match the teacher's output; the teacher is a slowly updated average of the student. Because nothing tells the model what objects are *called*, its features capture what things *look like*. That is exactly the property image-to-image retrieval needs.

The flagship is a 6.7B-parameter ViT-7B trained on **LVD-1689M**, a 1.689B-image subset of about 17B public Instagram images. The subset was chosen by hierarchical clustering so that rare visual concepts are not swamped by common ones. Meta distilled the 7B teacher into smaller models ([DINOv3](https://arxiv.org/html/2508.10104v1); [GitHub](https://github.com/facebookresearch/dinov3)):

- ViT students: S (21M), S+ (29M), B (86M), L (300M) and H+ (840M);
- four ConvNeXt students;
- a satellite variant trained on 493M Maxar images.

Data curation matters a lot. At 200k iterations, Paris retrieval scored 63.3 on raw data and 85.9 on the curated set ([DINOv3](https://arxiv.org/html/2508.10104v1)).

The paper's main technical contribution is **Gram anchoring**. Over very long training, DINO-style patch features degrade: patches on different objects start to look alike. The fix works like this:

1. Take an image and compute a feature for every patch.
2. Fill a table with how similar each patch is to every other patch (a "Gram matrix").
3. Compute the same table with an earlier teacher checkpoint whose patch features were still crisp.
4. Penalize the student whenever its table drifts from the earlier one.

The student keeps improving its global understanding but must preserve the earlier model's sense of which patches belong together. For a photo of a beetle on a leaf, the table should keep beetle-to-beetle similarity high and beetle-to-leaf similarity low. If a long training run lets beetle-to-leaf similarity creep up, the loss pushes it back down.

Meta switched this loss on after 1M iterations and refreshed the reference checkpoint every 10k iterations. ADE20k segmentation rose from 50.3 to 53.6 mIoU while ImageNet linear accuracy stayed flat (88.2 to 88.0) ([DINOv3](https://arxiv.org/html/2508.10104v1)). In plain terms, the patch features got much better at no cost to global quality. Patch features stay stable at high resolution, from 896×512 up to 3584×2048 for ViT-S+, while ViT-L starts to drift at 7168×4096 ([DINOv3](https://arxiv.org/html/2508.10104v1)).

**License.** Weights ship under a custom DINOv3 License. It grants a royalty-free, worldwide license with no user-count threshold. It bans ITAR-controlled, military, nuclear and espionage uses, and requires acknowledgment when research using the materials is published ([LICENSE](https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md)). Meta's blog calls it a "commercial license", but the license text never says "commercial" explicitly ([Meta blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)). Anyone deploying it in a dual-use domain should get legal review.

### Same-size comparisons show a 40-point lead on instance retrieval

The cleanest comparison is the paper's table of distilled models against same-size competitors. ObjectNet and Oxford-Hard measure global and retrieval quality; ADE20k measures dense, patch-level quality.

| Size | Model | ImageNet-ReaL | ObjectNet | Oxford-Hard retrieval | ADE20k |
| --- | --- | --- | --- | --- | --- |
| B | SigLIP 2 | 89.3 | 66.9 | 20.2 | 41.6 |
| B | DINOv2 | 89.0 | 57.3 | 51.0 | 48.4 |
| B | DINOv3 | 89.3 | 64.1 | **58.5** | 51.8 |
| L | PEcore | 90.1 | 74.9 | 25.6 | 39.7 |
| L | SigLIP 2 | 90.1 | 75.0 | 21.4 | 43.6 |
| L | DINOv2 | 89.7 | 64.7 | 55.7 | 48.8 |
| L | DINOv3 | 90.2 | 74.8 | **63.1** | 54.9 |
| SO400m | SigLIP 2 | 90.3 | 76.2 | 23.0 | 44.0 |
| H+ | DINOv3 | 90.3 | 78.6 | **64.5** | 54.8 |

Source: [DINOv3](https://arxiv.org/html/2508.10104v1).

Classification accuracy is a near tie. On Oxford-Hard instance retrieval, DINOv3 leads SigLIP 2 and PE by about 40 points at the same size, and DINOv2 by 7 points. At the largest scale, DINOv3-7B scores **Met 55.4 against 44.6** for DINOv2-g and AmsterTime 56.5 against 48.9 ([DINOv3](https://arxiv.org/html/2508.10104v1)).

**Comparability:** the 7B model scores 60.7 on Oxford-Hard in the paper's large-model table, but the distilled L and H+ score 63.1 and 64.5 in the same-size table. The two tables almost certainly use different resolutions or protocols, so compare models only within one table. In the large-model table, row labels other than DINOv3 and DINOv2 could not be verified from the fetched excerpt, so its CLIP-family numbers are not quoted.

DINOv3 is not best at everything. On fine-grained linear classification, PEcore-G beats DINOv3-7B on the average (94.5 vs. 93.0) and on iNat18, while DINOv3 leads on iNat21 ([DINOv3](https://arxiv.org/html/2508.10104v1)).

**Inference:** the distilled ViT-L/16 or ViT-H+/16 is the practical retrieval backbone. H+ costs 1,903 GFLOPs at 512 px against 14,515 for the 7B ([DINOv3](https://arxiv.org/html/2508.10104v1)), so it is about 7.6× cheaper for nearly the same retrieval quality. VR-RAG's reranker used DINOv2 ([VR-RAG](https://arxiv.org/html/2505.05635v2)); swapping in DINOv3 is the obvious first experiment for exemplar reranking.

### Text queries need a text-aligned partner encoder

DINOv3 has no native text tower. Meta's official add-on, **dino.txt**, trains a text encoder plus two adapter layers against the *frozen* vision model, in the style of LiT. It aligns text with a concatenation of the global token and the average of the patch tokens. Results at ViT-L size:

| Task | dino.txt | SigLIP 2 | PE |
| --- | --- | --- | --- |
| COCO text-to-image R@1 | 45.6 | 55.3 | 57.1 |
| Zero-shot open-vocabulary segmentation, ADE20k (mIoU) | **24.7** | 10.8 | 17.6 |

Source: [DINOv3](https://arxiv.org/html/2508.10104v1).

So text-to-image search is weaker, but text-*to-region* grounding is stronger. A controlled study trained CLIP and DINO with matched architecture and data. CLIP captured high-level semantics and text, and won text-heavy VQA. DINO was more sensitive to low-level cues like color and style, and won vision-centric VQA by a small margin ([CLIP vs DINO](https://arxiv.org/abs/2510.11835)). Fusion models that combine both encoders report gains over either alone: CoME-VL reports +4.9% on understanding and +5.4% on grounding ([CoME-VL](https://arxiv.org/abs/2604.03231)).

**Inference:** index two global embeddings per image:

- **DINOv3** for "find images that look like this";
- **SigLIP 2 or PE** for free-text queries and for images whose printed text matters (labels, charts).

Because dino.txt-style alignment freezes the vision model, retraining the text side on domain captions never forces re-embedding the image corpus. That is a valuable property for a RAG index that keeps growing.

### Patch features make part- and region-level retrieval possible

Gram-anchored patch tokens are DINOv3's differentiator, and 2025–26 systems already use them as retrievers:

- **SegRAG** keeps class-indexed banks of DINOv3 ViT-L patch descriptors from annotated reference images. A 1536×1536 input gives 96×96 patches of 1,024 dimensions. Each query patch is scored by its best cosine match in a class bank, and the peaks become point prompts for SAM 3. On agricultural AgML collections, mIoU rose from **25.27 to 59.24** with up to 30 shots. Cauliflower went from 0.00 IoU, where text-only SAM 3 failed completely, to 95.36. The authors say gains are largest under domain shift and modest on in-distribution benchmarks ([SegRAG](https://www.alphaxiv.org/abs/2605.17630.md)).
- **Memory-SAM** retrieves the nearest annotated exemplar using dense DINOv3 features through FAISS. It turns patch correspondences into prompts and reaches 0.9863 mIoU on tongue images from 600 expert annotations ([Memory-SAM](https://arxiv.org/abs/2510.15849)).
- **FSSDINO** reports a "semantic selection gap": some intermediate layers beat the last layer, but no tested heuristic finds them reliably without an oracle ([FSSDINO](https://arxiv.org/abs/2602.07550)).

**Inference:** a natural two-stage retriever is approximate nearest-neighbor search over DINOv3 global vectors, followed by a patch-level MaxSim rerank. In the rerank, each query patch finds its best match among the candidate's patches, and the image's score is the average of those best matches. To restrict it to the question's region, rerank only patches inside the grounded box. SegRAG's operator is exactly this, applied to class banks. No paper benchmarks it for DINOv3 image-to-image retrieval or full image-QA RAG.

Patch banks are large: 9,216 vectors of 1,024 dimensions per image at 1536 px. The vector-index section below shows why projection and merging are mandatory before indexing them.

### Adapting DINOv3 to a domain: a four-rung ladder

Frozen DINOv3 is a strong baseline even far from natural photos:

- **Medicine.** A benchmark across 2D/3D classification, segmentation and registration found frozen DINOv3 beating medical foundation models such as BiomedCLIP on several tasks. It degraded on whole-slide pathology, electron microscopy and PET, and did "not consistently obey scaling laws": bigger models and finer resolution did not reliably help ([medical benchmark](https://arxiv.org/abs/2509.06467)).
- **Pathology.** LoRA with about 1.3M trainable parameters on DINOv3-H+ won the MIDOG 2025 atypical-mitosis challenge ([MIDOG](https://arxiv.org/abs/2508.21041)).
- **Continued pretraining.** NVIDIA TAO supports continued self-supervised pretraining from DINOv3 weights on unlabeled domain images, with optional Gram anchoring and an option to keep the global token near a frozen teacher ([NVIDIA TAO](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/self_supervised_learning/dinov3.html)).
- **Remote sensing.** Meta retrained on satellite imagery, yet the web model still won several high-resolution remote-sensing tasks: LoveDA, DIOR, and the GEO-Bench means ([DINOv3](https://arxiv.org/html/2508.10104v1)).

**Inference:** climb only as far as the domain requires:

1. Frozen ViT-L/H+ with a k-NN evaluation on a small labeled set.
2. A small metric-learning projection on frozen features. Retraining it only means re-projecting stored features, not re-encoding images.
3. LoRA on the backbone.
4. Continued self-supervised pretraining with Gram anchoring and global-token preservation, so the general prior is not wrecked.

Natural-image-like domains (crops, parts, retail, endoscopy) usually stop at rung 1 or 2. Modality-shifted domains (whole slides, EM, PET, non-RGB sensors) are where rungs 3–4 are likely needed. Tune model size and resolution per domain rather than defaulting to 7B at maximum resolution.

### Known weaknesses and the missing experiment

The documented weaknesses:

- weak coarse semantics relative to CLIP ([Granulon](https://arxiv.org/abs/2603.08800));
- failure on far-domain modalities;
- unreliable layer selection;
- slight low-resolution degradation after high-resolution adaptation;
- very high compute for the 7B: 61,440 H100 GPU-hours to pretrain ([DINOv3](https://arxiv.org/html/2508.10104v1)).

The largest gap in the evidence: as of November 2026, the research found **no paper using DINOv3 as the retriever in a full image-QA RAG pipeline with an LLM generator and reported QA accuracy**. The case for DINOv3 in this role rests on retrieval benchmarks and retrieve-then-segment systems, not on end-to-end QA.

## Where diffusion models fit

Diffusion models earn their place offline, as generators of hard negatives and counterfactual training pairs. Online, they are slow (12–27 s per round), often hallucinate niche concepts, and lose to simply retrieving real images.

### Ranking the roles by evidence

| Role | Strongest evidence | Online cost | Verdict |
| --- | --- | --- | --- |
| Synthetic training data | Real retrieval beats it on fine-grained tasks; domain-adapted generators help medicine | None (offline) | Use only after a real-retrieval baseline, and target the gaps |
| Hard negatives and counterfactual pairs for retriever training | +9% SugarCrepe; +33% Flickr30k-Positions | None (offline) | Most reliably positive use, though not yet shown in niche domains |
| Diffusion features | Beat DINOv2-B on part correspondence; fusion helps | Full UNet pass at \~960 px | Second-stage part verification only |
| Imagine-then-retrieve | Large gains on natural-image interactive retrieval | 12–27 s per round | Gate it to hard, text-only queries |
| Illustrations and counterfactual explanations | Established methods; no QA user studies | Seconds to minutes | Offline galleries or opt-in; always label as synthetic |
| Restoring the query before retrieval | Fidelity gains with unchanged hallucination rates | Seconds | Avoid; if used, retrieve with raw and restored queries side by side |

The subsections below give the sources behind each row.

### Synthetic data must beat a retrieval baseline first

The sharpest result is Geng et al. (NeurIPS 2024): Stable Diffusion data "is universally matched or outperformed by real data from the simple retrieval baseline" ([Geng et al.](https://arxiv.org/html/2406.05184v2)).

- On FGVC-Aircraft, 139K synthetic images lifted a CLIP linear probe from 64.9% to 68.7%, while 139K retrieved LAION images lifted it to **82.7%**.
- About 500K synthetic images were needed to match 15K retrieved ones.
- The causes: generator artifacts and wrong class-specific details, such as the wrong engine and wheel layout for an A320.

Generated data does help when the generator is adapted to the domain:

- RoentGen, a latent diffusion model adapted to chest X-rays and reports, improved a classifier by 5% when synthetic and real data were combined ([RoentGen](https://arxiv.org/abs/2211.12737)).
- Label-conditioned generative models improved out-of-distribution accuracy by 7.7% in histopathology and 5.2% in chest X-ray ([Ktena et al.](https://arxiv.org/abs/2304.09218)). The arXiv abstract does not explicitly say these were diffusion models.
- Few-shot personalization methods such as DA-Fusion and Diff-Mix report gains on domain-specific and long-tail classification ([DA-Fusion](https://arxiv.org/abs/2302.07944); [Diff-Mix](https://arxiv.org/abs/2403.19600)).

For training retrievers, hard negatives are the most reliably positive use:

- **TripletCLIP** has an LLM write hard-negative captions and a text-to-image model render matching negative images. It gains over 9% absolute on SugarCrepe ([TripletCLIP](https://arxiv.org/abs/2411.02545)).
- **CounterCurate** builds positional counterfactuals with GLIGEN. It lifts CLIP by about 33% and LLaVA by about 37% on Flickr30k-Positions ([CounterCurate](https://arxiv.org/abs/2402.13254)).

One long-run warning: self-consuming loops that retrain on generated data without enough fresh real data lose quality or diversity ([MAD](https://arxiv.org/abs/2307.01850)).

**Inference:** in a niche domain, the main hazard is a retriever learning generator quirks instead of diagnostic features. Generate counterfactual pairs that differ in exactly the discriminative trait ("same beetle, spots removed"). Filter them with a domain classifier or an expert, and evaluate only on held-out real images.

### Diffusion features: strong for parts, unproven for global search

Text-to-image diffusion networks contain surprisingly good dense features:

- **DIFT** beats DINO by 19 points and OpenCLIP by 14 on SPair-71k semantic correspondence, without any fine-tuning ([DIFT](https://arxiv.org/abs/2306.03881)).
- **Fusing Stable Diffusion with DINOv2** features gives 64.0 zero-shot PCK@0.10 on SPair-71k, against 57.2 for SD alone and 55.6 for DINOv2-B alone. The authors describe DINOv2 as giving "sparse but accurate matches" and SD as giving good spatial layout but sometimes wrong semantics. Running SD "considerably increases" compute at 960×960 input ([A Tale of Two Features](https://arxiv.org/html/2305.15347v2)).
- **Global quality lags.** The REPA paper states that diffusion models' internal representations still lag behind self-supervised encoders ([REPA](https://arxiv.org/abs/2410.06940)).
- **Diffusion classification is slow.** It needs one denoising pass per class per noise sample, about 18 s per image even on the 37-class Pets dataset ([Diffusion Classifier](https://ar5iv.labs.arxiv.org/html/2303.16203)).
- **An indirect route.** DIVA uses diffusion reconstruction feedback to post-train CLIP's image encoder. Fine-grained perception rises 3–7% on MMVP-VLM with no diffusion in the online path ([DIVA](https://arxiv.org/abs/2407.20171)).

**Inference:** use diffusion features, if at all, as a second-stage part- or geometry-verification signal on the top few candidates. No benchmark shows diffusion global descriptors competing with DINO on fine-grained category or instance retrieval.

### Imagine-then-retrieve works for text queries if gated

The idea: turn a text query into a picture, then search image-to-image. Worked example: a user types "black beetle with four orange spots on the wing covers" against a gallery indexed with DINOv3. Text-to-image search through a CLIP tower crosses the modality gap. Instead, a diffusion model renders a prototype beetle, DINOv3 embeds it, and the system searches image-to-image. The proxy's score is fused with the text-search score rather than replacing it.

The results are strong on natural-image benchmarks:

- **GenIR** renders the query each round of a dialogue. It reaches about 98% Hits@10 on MSCOCO by round 10, against 73% for ChatIR ([GenIR](https://papers.neurips.cc/paper_files/paper/2025/file/c87da7d6ba0fb37236df181db9c2b8ff-Paper-Conference.pdf)).
- **IP-CIR** raises composed image retrieval CIRCO mAP@10 from 32.24 to 34.26 ([IP-CIR](https://arxiv.org/abs/2411.16752)).

The costs are equally clear:

- **Latency.** GenIR costs 12–27 s per round versus about 2 s for text feedback. A gated hybrid that generates for about 22% of queries gains about 3.4 points for about 3 s extra on average ([GenIR](https://papers.neurips.cc/paper_files/paper/2025/file/c87da7d6ba0fb37236df181db9c2b8ff-Paper-Conference.pdf)).
- **Hallucinated proxies.** In DMCL (SIGIR 2026), about 40% of SD 3.5 proxies were judged inconsistent with the dialogue. An untrained fusion baseline underperformed text-only retrieval in early rounds. DMCL's trained multi-view fusion, which learns to down-weight the diffusion view, recovered +4.90 to +7.37 Hits@10 ([DMCL](https://arxiv.org/pdf/2601.20391)).

**Comparability:** DMCL's 40% figure comes from AI judges that disagreed with each other (56.9% vs. 51.3% matched), and the authors flag such ratios as error-prone. Every positive result here comes from MSCOCO, FFHQ, fashion or CIRR, which are close to the generators' training data.

**Inference:** in a niche domain the generator must be able to draw the concept. For a rare species, a defect type, or an imaging modality like histology, proxies will often be wrong. Use them as a gated, down-weighted extra view, triggered only when text-to-image retrieval confidence is low.

### Illustrations and counterfactuals, never restored evidence

Diffusion can also run in the other direction, conditioning a generator on retrieved images:

- RDM swaps its retrieval database to move to a new domain without retraining ([RDM](https://arxiv.org/abs/2204.11824)).
- Re-Imagen improves fidelity for rare entities ([Re-Imagen](https://arxiv.org/abs/2209.14491)).
- ImageRAG is training-free and reports gains on rare and fine-grained concepts ([ImageRAG](https://arxiv.org/abs/2502.09411)).

These could illustrate an answer, such as "what a typical lesion of type X looks like." Diffusion counterfactuals can also show "the minimal change that would flip this to class B." ACE uses diffusion as a regularizer to make adversarial edits meaningful ([ACE](https://arxiv.org/abs/2303.09962)). Atad et al. generate ordinal counterfactuals for vertebral fractures and diabetic retinopathy in a diffusion-autoencoder latent space ([Atad et al.](https://arxiv.org/abs/2408.01571)).

Restoring a low-quality query image before retrieval is the risky case. In medical super-resolution, swapping the generic SD autoencoder for a medical one gave +2.91 to +3.29 dB PSNR but **left hallucination rates unchanged** ([MedVAE SR](https://arxiv.org/abs/2604.12152)). A 2026 sampling method improved PSNR while worsening FID, and its authors stress it "is not a hallucination detector" ([RPU](https://www.alphaxiv.org/abs/2606.02331.md)).

**Inference:** in fine-grained domains, the plausible-but-invented details that restoration adds can be exactly the features that decide which neighbors come back. If you must restore, retrieve with both the raw and the restored query and compare the results. Never show restored content as observed evidence. Pair any counterfactual with real retrieved exemplars of the alternative class, and mark generated images as synthetic in the interface.

### Unified understand-and-generate models trade understanding for convenience

- **Janus** separates the encoders for understanding and generation, because one shared encoder hurts understanding ([Janus](https://arxiv.org/abs/2410.13848)).
- **BLIP3-o** trains understanding first, then generation, to preserve understanding ([BLIP3-o](https://arxiv.org/abs/2505.09568)).
- **Transfusion** trains one transformer with a next-token loss on text and a diffusion loss on images ([Transfusion](https://arxiv.org/abs/2408.11039)).
- **OmniBridge** adds retrieval to the same model. It reaches 92.5 average recall zero-shot on Flickr30K, but DocVQA drops from 94.5 to 92.4 and InfoVQA from 76.4 to 72.9 relative to its base model ([OmniBridge](https://arxiv.org/html/2509.19018v1)).

**Inference:** for niche QA, where understanding accuracy matters most, a strong understanding-only VLM plus a separate, swappable generator is the lower-risk default today.

### Cost and evaluation checklist

Online generation is the most expensive component it enters. Full-size generators take about 20 s per 1024² image on an A6000 ([GenIR](https://papers.neurips.cc/paper_files/paper/2025/file/c87da7d6ba0fb37236df181db9c2b8ff-Paper-Conference.pdf)). Adversarial Diffusion Distillation reaches SDXL-level quality in 4 steps ([ADD](https://arxiv.org/abs/2311.17042)), but its fidelity on niche concepts is unmeasured. Generators also memorize: Carlini et al. extracted over 1,000 training images from large diffusion models ([Carlini et al.](https://arxiv.org/abs/2301.13188)), so "synthetic" evaluation data can leak test items.

**Inference:** any diffusion-augmented system should report:

- a "retrieve real images instead" baseline;
- results on held-out real images only;
- a near-duplicate check of generated images against the test set, using retrieval-based replication detection ([Somepalli et al.](https://arxiv.org/abs/2212.03860));
- human spot checks of proxy consistency;
- per-query latency and memory;
- answer changes caused by restoration, not only PSNR or FID.

## Adaptive and agentic retrieval control

A cheap gate that skips retrieval for perception questions captures most of the benefit. Full RL search agents pay off only when multi-hop, open-web lookup is common, and they need a search penalty or they learn to always search.

### Text RAG split the retrieval decision into four questions

1. **Whether** to retrieve at all.
2. **When**, during generation, to retrieve.
3. **What** query to send.
4. **Whether the results are good enough** to use.

Each method attacks one or more of these. Self-RAG emits reflection tokens to retrieve on demand and to critique passages and its own output ([Self-RAG](https://arxiv.org/abs/2310.11511)). CRAG scores the retrieved set and switches to web search when confidence is low ([CRAG](https://arxiv.org/abs/2401.15884)). FLARE retrieves when the upcoming sentence has low-confidence tokens ([FLARE](https://arxiv.org/abs/2305.06983)), and DRAGIN does so when it detects a real-time information need ([DRAGIN](https://arxiv.org/abs/2403.10081)). SKR answers questions the model "knows" internally, because retrieval sometimes hurts answers it would otherwise get right ([SKR](https://arxiv.org/abs/2310.05002)).

Adaptive-RAG shows how much depends on routing quality. A T5-Large classifier routes each question to one of three strategies: no retrieval, single-step, or multi-step. Its labels are generated automatically: run all three strategies on training questions, and give each question the cheapest strategy that answered it correctly. When none did, fall back to the dataset's type: single-hop datasets get single-step, multi-hop datasets get multi-step.

The real classifier is only **54.52% accurate**, and just 30.52% on the "no retrieval" class. With GPT-3.5 it reaches 50.91 F1 at 1.46× single-step time. An oracle router reaches **62.80 F1 at 1.03×**: more accurate *and* cheaper, because easy questions stop taking expensive paths ([Adaptive-RAG](https://arxiv.org/pdf/2403.14403v2)).

Moskvoretskii et al. (ACL 2025) benchmarked 27 uncertainty-estimation triggers against these pipelines ([Moskvoretskii et al.](https://arxiv.org/html/2501.12835v1)):

- On TriviaQA, **always retrieving (0.610) scored below never retrieving (0.636)**, and the best uncertainty trigger reached 0.662 with only 0.28 retrieval calls per question.
- Complex methods still won on multi-hop questions: on 2Wiki, DRAGIN scored 0.456 against 0.384 for the best uncertainty trigger.
- Most adaptive methods' decisions barely correlated with whether the model actually knew the answer, so QA accuracy is a poor proxy for decision quality.

Probing-RAG uses a 5 MB probe on Gemma-2B's hidden states. It skipped retrieval on 57.46% of questions and beat every baseline (35.8 vs. 29.2 for no retrieval), with 1,988 retrieval calls against DRAGIN's 13,570 ([Probing-RAG](https://arxiv.org/html/2410.13339v2)).

### Images make the "don't retrieve" branch matter more

Multimodal evidence resolves an apparent contradiction. ReflectiVA found that forcing retrieval barely changed E-VQA (35.5 vs. 35.3) ([ReflectiVA](https://arxiv.org/abs/2411.16863)). On mixed workloads, by contrast, gating matters a great deal.

Chen et al. labeled each training query by sampling the VLM's answer 30 times and averaging a judge's 1–5 scores. They then LoRA-tuned the VLM to predict that score, giving a "soft knowledge boundary." Results with Qwen-VL-7B ([knowledge boundary](https://arxiv.org/html/2502.18023v3)):

- On a mixed set, it retrieved 49.33% of the time and scored 39.93, against 38.60 for always-retrieve and 34.44 for never.
- On NoCaps, always-retrieve collapsed accuracy from **50.13 to 30.37**, while the boundary model retrieved 0% of the time and kept 50.13.
- Time to answer fell 30.90%.
- The boundary transferred to GPT-4o (52.97 vs. 52.90 for always-retrieve).

Which confidence signal to trust also matters. A VLM's *verbalized* confidence is overconfident. Answer consistency across samples aligned at 0.898 on Dyn-VQA for LLaVA-1.5, against 0.534 for verbalized confidence ([Ding et al.](https://arxiv.org/pdf/2508.19111)). Fixed retrieval pipelines also hurt strong models: a heuristic multimodal RAG lowered GPT-4V's InfoSeek score from 70.44 to 58.64 ([OmniSearch](https://arxiv.org/html/2411.02937v5)).

**Inference:** when-to-retrieve barely matters on knowledge-only benchmarks, because nearly every question needs retrieval. It matters a lot on real workloads that mix "what species is this?" with "how many are there?". The gating recipe:

1. Sample the VLM 5–10 times, or use answer-token probability.
2. If the samples agree and the question is perceptual (count, color, position), answer directly or zoom.
3. Otherwise retrieve, then run each retrieved item through a relevance gate.

### Three control architectures with very different costs

| Architecture | Examples | Data and compute | Key result |
| --- | --- | --- | --- |
| Reflective VLM: special tokens gate retrieval and relevance in one model | ReflectiVA, mR²AG | \~85K GPT-4-labeled samples, then \~6M SFT samples (ReflectiVA) | E-VQA single-hop 35.5 vs 16.3 for LLaVA-1.5-7B without retrieval ([ReflectiVA](https://arxiv.org/abs/2411.16863)) |
| Planner plus tools: an agent writes sub-questions and picks an API per step | OmniSearch | Prompting, or LoRA on \~13K trajectories | Dyn-VQA F1-Recall 50.03 vs 37.15 for a two-step pipeline, at $0.0446 vs $0.0185 per question and 22–31 s latency ([OmniSearch](https://arxiv.org/html/2411.02937v5)) |
| RL search agent: GRPO teaches when and how to search | MMSearch-R1, DeepMMSearch-R1, VSearcher, PMSR | 1K–10K QA pairs, but live-search rollouts on multi-node H100 clusters plus an LLM judge | MMSearch-R1 scores 54.6 searching 67.1% of the time, vs 51.6 for always-RAG at the same size ([MMSearch-R1](https://arxiv.org/pdf/2506.20670)) |

RL agent details:

- **Data balance and a search penalty are essential.** Without them, MMSearch-R1's search rate drifts to about 100% ([MMSearch-R1](https://arxiv.org/pdf/2506.20670)).
- **Crop-then-search helps.** DeepMMSearch-R1 adds a crop tool worth about +1.75 points. After RL it uses crops 35–37% less often while accuracy rises, and it uses tools on 87.7% of Dyn-VQA questions but only 43.5% of OK-VQA questions ([DeepMMSearch-R1](https://arxiv.org/html/2510.12801v1)).
- **Compact memory and adaptive stopping.** PMSR keeps short "records" instead of the full history and stops when consecutive queries become near-duplicates. Iterations fell from 5.0 to 3.3 on InfoSeek with recall of 94.6 vs. 95.0. Its persistent-failure rate on E-VQA was 4.9%, against 51.8% for WebWatcher ([PMSR](https://aclanthology.org/2026.acl-long.881.pdf)).
- **Context length limits agents.** In OmniSearch's error analysis, 73 of 100 failures had partly found the answer, but search stopped because the context grew too long ([OmniSearch](https://arxiv.org/html/2411.02937v5)).

**Comparability:** numbers across agent papers diverge sharply because search APIs, judges and splits differ. MMSearch-R1's SimpleVQA score is 53.90 in DeepMMSearch-R1's table and 40.0 in VSearcher's ([DeepMMSearch-R1](https://arxiv.org/html/2510.12801v1); [VSearcher](https://arxiv.org/html/2603.02795)). The judge alone moves results: MMSearch-R1 averaged 59.5% with GPT-4o as judge against 55.7% with exact match ([MMSearch-R1](https://arxiv.org/pdf/2506.20670)). Compare only within one paper's table.

### Zoom and crop are retrieval-like actions

For high-resolution detail, the most useful "retrieval" is often into the image itself:

- **DeepEyes** trains Qwen2.5-VL-7B with RL to call a zoom-in tool up to six times. V\* Bench rises from 71.2 to **90.1**, and HR-Bench 4K from 68.8 to 75.1. The tool reward made the difference: rewarding a zoom only when the final answer was correct gave 75.1 on HR-Bench 4K, against 53.4 with no tool bonus. Over training, use went from exploratory to frequent to selective ([DeepEyes](https://arxiv.org/pdf/2505.14362v3)).
- **ZoomEye** gets 15–18% HR-Bench gains for an 8B model with training-free tree search over zoomed sub-regions ([ZoomEye](https://arxiv.org/abs/2411.16044)).
- **sCoT** has a main agent split the question into region-bound subtasks. Each subtask goes to a weight-sharing subagent that sees only its crop in an isolated context. It reaches V\* 91.6 with about 75% fewer GPU hours than DeepEyes, on different hardware ([sCoT](https://arxiv.org/pdf/2512.08511v2)).
- **VRAG-RL** applies the same idea to retrieved pages: the agent crops and re-encodes regions of already-retrieved documents ([VRAG-RL](https://arxiv.org/html/2505.22019v2)).

### Routers are accurate in-domain and brittle outside it

UniversalRAG routes each query to no retrieval or to one of six modality and granularity corpora (paragraph, document, table, image, clip or video), each with its own index ([UniversalRAG](https://arxiv.org/pdf/2504.20734)):

- In-domain, a trained Qwen3-VL-2B router picks the right corpus 95.81% of the time. Routed retrieval scores 42.40, against about 33–34 for a single unified embedding space.
- Out of domain, the trained router falls to 71.29%. A prompted GPT-5 holds 77.38%, and a confidence ensemble of routers does best.

MMA-RAG predicts from VLM hidden states whether reverse image search will help. Its gains over always-retrieve were small, +0.1 to +3.4 points, and the useful signal sits partly in early-layer vision features ([MMA-RAG](https://arxiv.org/pdf/2603.00511)).

### A concrete action space for niche image QA

**Inference:** combining these designs gives six actions:

- **answer\_direct**;
- **zoom** for perception, with no external call;
- **ground-and-crop, then exemplar search**;
- **whole-image exemplar search**;
- **text knowledge-base or web search**;
- **stop**, using a similarity-based termination rule.

Worked example: "What species is the beetle in the lower left, and is it a protected species?" The agent zooms into the beetle, runs exemplar search on the crop, runs a text search for the species' status, and stops. "How many beetles are on the leaf?" goes to zoom or answer\_direct, with no retrieval.

Build the cheap layer first. A hidden-state or consistency gate, trained on a few thousand examples self-labeled by running the VLM with and without retrieval, costs little and captures most of the gating benefit. Use RL agents only if multi-hop, open-web lookup is common, and train them with 30–50% search-free questions plus a search penalty. Keep an LLM-router fallback for out-of-distribution question types.

## Vector indexes, quantization, and storage design

From 10^3 to 10^7 images, the choice of index is rarely the bottleneck. The real costs are patch-level multi-vectors, selective metadata filters, and re-embedding the corpus when encoders change.

### Below a million images, exact search is a serious option

Four index families matter here:

- **Flat (exact) search** compares the query against every vector.
- **HNSW** builds a multi-layer "highway" graph, with sparse long-range links on top and dense local links at the bottom. A query hops greedily toward closer neighbors.
- **IVF** clusters vectors into buckets and searches only the few buckets nearest the query.
- **DiskANN** keeps a graph on SSD with compressed vectors in RAM.

Some practical numbers:

- **Flat.** FAISS notes that only flat indexes guarantee exact results, and that flat is most efficient when you run only 1,000–10,000 searches, because build time is never recovered ([FAISS](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)).
- **HNSW.** It uses about (4 × dimension + 8 × M) bytes per vector, and FAISS's HNSW cannot delete vectors ([FAISS](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)).
- **DiskANN.** It serves a billion points from 64 GB of RAM at over 5,000 queries per second with 95%+ 1-recall@1 ([DiskANN](https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)).
- **Clustered indexes on cheap storage.** These suit disk and object storage and update without rebuilds, but give lower throughput than HNSW at comparable recall ([Weaviate HFresh](https://weaviate.io/blog/hfresh); [turbopuffer](https://turbopuffer.com/docs/architecture)).
- **GPUs speed up builds.** CAGRA built indexes 6.4× to 12.3× faster than CPU HNSW, and can hand an HNSW-compatible graph to CPU servers ([Meta Engineering](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/); [Milvus](https://milvus.io/blog/faster-index-builds-and-scalable-queries-with-gpu-cagra-in-milvus.md)). That matters when you re-embed the corpus after swapping encoders.

**Inference, worked arithmetic:**

- One million 1,024-dimension float32 vectors take about 4.1 GB. A full scan at the \~300 GB/s DRAM bandwidth turbopuffer cites ([turbopuffer](https://turbopuffer.com/blog/ann-v3)) costs about 14 ms in theory.
- At 10M images, the same vectors take about 41 GB in float32, 10 GB in int8, or 1.3 GB at 1 bit. An M=32 HNSW graph adds about 2.6 GB.
- So one 32–64 GB node can serve 10M images, with compressed codes in RAM and full-precision originals on SSD for rescoring.

| Global vectors | Suggested index (inference from FAISS guidance and vendor docs) |
| --- | --- |
| Under 100k | Flat exact search |
| 100k–1M | Flat, or HNSW with int8 |
| 1M–10M | HNSW over int8, 4-bit or binary codes with float rescoring; or IVF+HNSW with PQ or RaBitQ in FAISS |
| 10M–100M, RAM-limited | DiskANN-family or clustered on-disk indexes |

### Quantize, oversample, rescore

Quantization stores each vector with fewer bits. The evidence on retention comes from text embeddings:

- **int8** cuts storage 4× and keeps about 97% of retrieval quality, or about 99% with rescoring ([Hugging Face](https://huggingface.co/blog/embedding-quantization)).
- **Binary (1 bit per dimension)** cuts storage 32× and keeps 92.5% without rescoring and 96.45% with it on a large text model ([Hugging Face](https://huggingface.co/blog/embedding-quantization)).
- **Binary is strongly model- and dimension-dependent.** One model kept only 74.77% ([Hugging Face](https://huggingface.co/blog/embedding-quantization)). Qdrant found binary "lost significant data" below about 1,000 dimensions: a 512-d model kept 53–55% recall without rescoring and 71–91% with it ([Qdrant](https://qdrant.tech/documentation/manage-data/quantization/)).
- **RaBitQ** stores one bit per dimension plus a small correction. Its distance estimate comes with a provable error bound, and it beats PQ at half the code length ([RaBitQ](https://arxiv.org/abs/2405.12497v1)). turbopuffer reranks fewer than 1% of candidates with RaBitQ 1-bit codes ([turbopuffer](https://turbopuffer.com/blog/ann-v3)).
- **Matryoshka-trained embeddings** can be truncated, up to 14× smaller at the same ImageNet accuracy ([Matryoshka](https://arxiv.org/abs/2205.13147)).

The universal mitigation for lossy codes is to oversample: search with the compressed codes for 3–4× more candidates than needed, then rescore them with full-precision vectors. Qdrant, Vespa, pgvectorscale, Weaviate and turbopuffer all implement this natively.

**Inference:** no fetched source reports quantization retention for *image* embeddings (SigLIP, or DINOv2/v3 global or patch tokens). int8 or 4-bit with rescoring is the safe default. Binary must be validated on held-out image queries. Off-the-shelf DINOv3 and SigLIP checkpoints were not Matryoshka-trained as far as the research shows, so do not assume truncation works.

### Patch multi-vectors must be merged, compressed, and used only for reranking

Patch-level storage is the main cost:

- ColPali emits about 1,024 128-dimension vectors per page and ColQwen2 about 768 ([Light-ColPali](https://arxiv.org/html/2506.04997v1)).
- **Inference, arithmetic:** DINOv3 ViT-L/16 at 512 px gives 1,024 patches of 1,024 dimensions, about 2 MiB per image in fp16, or about 19 TiB for 10M images. At SegRAG's 1536 px it is about 18 MiB per image. Raw DINO patches lack ColPali's 128-dimension projection head, so a learned projection or PCA to about 128 dimensions, plus merging, is effectively mandatory. A 1,024-patch image merged 4× and binarized at 128 dimensions comes to about 4 KiB.

Three reductions stack:

- **Merging.** Light-ColPali found that pruning patches is a bad idea: random pruning beat score- and attention-based pruning, because the patches a query activates are unpredictable. Merging similar patches by semantic clustering keeps **98.2% of retrieval quality at 11.8% of memory** ([Light-ColPali](https://arxiv.org/html/2506.04997v1)). Token pooling by factor 3 cut vectors by two-thirds at under 1% loss on text ([token pooling](https://arxiv.org/html/2409.14683v1)), and ColPali at pool factor 3 keeps 97.8% ([Elastic](https://www.elastic.co/search-labs/blog/late-interaction-model-colpali-scale)). Text-dense pages degrade fastest.
- **Binarization**, for 32× compression.
- **A single-vector first stage.** MUVERA turns a bag of patch vectors into one long vector. A dot product between two of these long vectors approximates the MaxSim score, so an ordinary index does first-stage retrieval and exact MaxSim reranks the shortlist. Against PLAID it gave about 10% higher recall with 90% lower latency on BEIR ([MUVERA](https://arxiv.org/html/2405.19504v2)). How the long vector is built:
  1. Random hyperplanes chop the space into regions.
  2. Within each region, the query's patch vectors are summed and the document's are averaged.
  3. The region summaries are concatenated into one long vector.

Databases already support the "global vector first, multi-vector rerank" pattern:

- Qdrant recommends *not* building an HNSW graph over token vectors (`m=0` on the multivector field) and using them only to rerank a dense prefetch ([Qdrant](https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/)).
- Elasticsearch pairs a pooled `dense_vector` with `rank_vectors` MaxSim rescoring ([Elastic](https://www.elastic.co/search-labs/blog/late-interaction-model-colpali-scale)).
- Weaviate builds MUVERA in, cutting memory 70–80% on its test set at some recall cost ([Weaviate](https://weaviate.io/blog/muvera)).

**Comparability:** every multi-vector compression number above was measured on text (ColBERT) or visual documents (ColPali/ColQwen). None was measured on natural-image DINO or SigLIP patches, where redundancy patterns differ.

### Regions: separate points with parent pointers, or entity-level vector lists

**Separate region points.** Each crop or detected object is its own point, carrying a `parent_image_id`. Searches group by that ID so one image takes one result slot. Qdrant's group query does this server-side, with a lookup that attaches the parent record. Qdrant warns that "deduplicating the results yourself after the search does not fill the page" ([Qdrant grouping](https://qdrant.tech/course/beginners/module-6/grouping/)).

**Entity-level vector lists.** All of an image's region vectors are stored inside one entity and scored with MaxSim, so retrieval is natively image-level. Milvus's Array of Structs does this ([Milvus](https://milvus.io/blog/unlocking-true-entity-level-retrieval-new-array-of-structs-and-max-sim-capabilities-in-milvus.md)).

**Inference:** use separate points when you need region-level filters (class label) or the bounding box returned directly. Use entity-level lists when answers are image-level and regions per image are bounded. Twenty regions per image across 10M images makes 200M points, so the region index, not the image index, sets the hardware tier.

### Selective filters break graph indexes unless planned for

Metadata filters like species, site, date, modality or license are often highly selective, and graph indexes handle them badly. If only 1% of images match "collected at site 14 in 2024", most of a node's graph neighbors fail the filter, and the greedy walk has nowhere valid to go. Qdrant measured this on 1M vectors:

| Filter selectivity | Plain graph | Plain graph + ACORN | Filterable HNSW |
| --- | --- | --- | --- |
| 10% match | 20.6% recall | 98.1% | 99.0% |
| 1% match | **0.1%** | 67.7% @ 4.7 ms | 99.8% @ 1.0 ms |

Source: [Qdrant](https://qdrant.tech/articles/filtered-vector-search-acorn/).

There are three families of fix:

- **Filter-aware edges added at index time.** These cost Qdrant 4.4–5.6× build time ([Qdrant](https://qdrant.tech/articles/filtered-vector-search-acorn/)).
- **Two-hop expansion at search time** (ACORN). The paper claims 2–1,000× throughput at fixed recall ([ACORN](https://arxiv.org/abs/2403.04871)).
- **Planners that fall back to exact search** when few items match. Examples are Qdrant's planner, pgvector 0.8's iterative scans and planner ([pgvector 0.8](https://www.postgresql.org/about/news/pgvector-0.8.0-released-2952/)), and HFresh's exact search for allow-lists under 5,000 IDs ([HFresh](https://weaviate.io/blog/hfresh)).

The NeurIPS'23 filtered-search track, on 10M CLIP image vectors, was won by per-tag graphs at 37,902 queries per second against 3,033 for the FAISS baseline. The authors warn that the one-or-two-tag setup may favor specialized structures ([Big-ANN 2023](https://arxiv.org/html/2409.17424v1)).

### Hybrid fusion and named vectors

**Inference:** store DINOv3, SigLIP 2 and caption-text vectors as named vectors on the same point. Qdrant supports several named vectors with different sizes and index settings in one collection ([Qdrant](https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/)). Run one prefetch per vector plus BM25 on captions and OCR, then fuse with reciprocal-rank fusion (RRF). Qdrant now makes the RRF constant configurable ([Qdrant 1.16](https://qdrant.tech/blog/qdrant-1.16.x/index.md)).

RRF in plain terms: each list contributes 1 / (constant + rank) for each item, and the item's scores are summed across lists. The conventional constant is 60. Worked example: image A ranks 1st in DINOv3 search and 8th in BM25, so it scores 1/61 + 1/68 ≈ 0.0311. Image B ranks 3rd in both, so it scores 2/63 ≈ 0.0317. B edges ahead because it is consistently good. RRF needs no score calibration, which matters when cosine scores from different encoders are not comparable. No source quantified RRF against weighted fusion for image-plus-caption retrieval.

### Choosing a database by workload, not leaderboard

As of November 2026, native multi-vector MaxSim exists in Qdrant, Weaviate (with MUVERA), Milvus (2.6.4+ and 3.0), LanceDB (cosine only), Vespa (tensors) and Elasticsearch (`rank_vectors`). The Pinecone, Chroma and turbopuffer docs fetched for this research do not mention it ([Weaviate](https://docs.weaviate.io/weaviate/configuration/compression/multi-vectors); [Milvus](https://milvus.io/docs/search-with-embedding-lists.md); [LanceDB](https://docs.lancedb.com/search/multivector-search); [Vespa](https://docs.vespa.ai/en/rag/binarizing-vectors.html); [Pinecone](https://docs.pinecone.io/guides/index-data/dedicated-read-nodes); [Chroma](https://docs.trychroma.com/cloud/schema/index-reference.md)).

| Scale | Viable choices (inference) | Design notes |
| --- | --- | --- |
| ≤100k images | LanceDB embedded, pgvector, Chroma, Qdrant single node, FAISS Flat + metadata store | Exact global search; brute-force MaxSim over the shortlist |
| 100k–5M | Qdrant, Weaviate or Milvus single node; Postgres + pgvectorscale if metadata lives in Postgres | Quantized HNSW with rescoring, named vectors, indexed filters, BM25 + RRF, rerank-only multivectors |
| 5M–50M images, or 100M+ region points | Milvus (GPU builds), Vespa (phased ranking), Qdrant or Weaviate with on-disk originals or HFresh; turbopuffer if \~0.5–1 s cold queries are acceptable | Region index sets the hardware tier |

**Comparability:** nearly all cross-database numbers are vendor-run, including pgvectorscale's claim of 28× lower p95 latency than Pinecone ([Tiger Data](https://www.tigerdata.com/docs/reference/pgvectorscale)) and Zilliz's VectorDBBench ([VDBBench](https://zilliz.com/blog/vdbbench-1-0-benchmarking-with-your-real-world-production-workloads)). Use them for mechanism insights, such as how badly filters hurt recall, not for ranking vendors. No public benchmark measures multi-vector, image-patch or region workloads at the database level.

**Inference:** run a small in-house benchmark covering:

- recall against exact search for each named vector under int8 and binary codes with rescoring;
- filtered recall at your domain's real selectivities;
- MaxSim rerank latency against merge factor;
- ingest and delete behavior when re-embedding with a new model.

## Research ideas and open problems

The gaps cluster where this field meets LLM evaluation and multi-agent reasoning. The table lists ten open problems with a first experiment for each; the paragraphs under it give the evidence behind each gap.

| Open problem | Why it is open | First experiment |
| --- | --- | --- |
| Label copying in exemplar RAG | Measured only in medical report RAG and general ICL | Label-swap and image-swap conditions across 3 domains |
| Visual claim attribution | Best grounding F1 0.089 vs 0.602 human | Claim-to-exemplar-region attribution metric with human validation |
| Per-candidate verifier agents | Decomposition used, never ablated against copying or position bias | Long-context vs isolated verifiers vs debate, fixed retrieval |
| Calibrated abstention with gating | No risk-coverage curves for VLM RAG | Conformal-style "none match" with RULE-style k calibration |
| DINOv3 as RAG retriever | No end-to-end QA evaluation exists | Global vs patch-MaxSim vs region-MaxSim vs SigLIP 2 |
| Stacked component gains | Never measured jointly | Additive ablation under one retriever and generator |
| Image-embedding quantization | Only text and document evidence | Retention of int8, binary, RaBitQ and merging on DINOv3/SigLIP |
| Diffusion hard negatives in niche domains | Only compositional benchmarks | Trait-flipped pairs vs retrieval-only baseline |
| Restore-then-retrieve | No retrieval-level study | Raw vs restored queries, neighbor overlap and accuracy |
| Gallery size and curation | Never quantified | Accuracy vs exemplars per class and expert-hours |

**Measuring whether answers come from the image or the neighbors.** Copy tests and over-reliance ratios exist only for medical report RAG and general in-context learning. Nobody has directly measured "label copying" in exemplar-gallery classification ([MMed-RAG](https://www.arxiv.org/pdf/2410.13085); [Baldassini et al.](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Baldassini_What_Makes_Multimodal_In-Context_Learning_Work_CVPRW_2024_paper.pdf)). A benchmark across several niche domains could fix this. It would apply the five controlled conditions plus a "label-swap" condition: keep the exemplar images but permute their labels. It would report flip counts and position sensitivity as first-class metrics. Claim-level visual attribution is wide open too. ViDoRe V3's grounding F1 of 0.089 against 0.602 human agreement shows that even frontier models cannot say which retrieved image supports which claim ([ViDoRe V3](https://arxiv.org/abs/2601.08620)). A metric for "this sentence is supported by exemplar 3, region (x, y)" would be the multimodal analogue of citation-faithfulness work in text RAG.

**Per-candidate verification agents.** The best systems already decompose by candidate:

- Patho-AgenticRAG splits a query into one sub-task per candidate diagnosis ([Patho-AgenticRAG](https://www.alphaxiv.org/overview/2508.02258)).
- sCoT shows isolated-context subagents matching end-to-end RL at a fraction of the training compute ([sCoT](https://arxiv.org/pdf/2512.08511v2)).

The open hypothesis: one verifier per candidate class can reduce copying and position bias. Each verifier would see only the query, its own exemplars and trait text in a fresh context, and the verifiers would argue before a judge decides. A single long context invites the U-shaped position bias and majority voting over neighbors ([position bias](https://www.emergentmind.com/papers/2506.11063)). The clean experiment holds retrieval fixed and compares four setups: one long-context VLM, per-candidate isolated verifiers, verifiers with a debate round, and shuffle-and-ensemble. Score genus- and species-level accuracy, plus the over-reliance ratio.

**Calibrated abstention combined with adaptive retrieval.** No fetched multimodal paper evaluates abstention together with adaptive retrieval or reports risk-coverage curves for VLM RAG. RULE's statistical calibration of k ([RULE](https://arxiv.org/html/2407.05131v2)) and the soft knowledge boundary ([knowledge boundary](https://arxiv.org/html/2502.18023v3)) are the ingredients. A "none of these candidates" verdict with a guaranteed error rate, plus a calibrated escalate-to-expert path, would turn DeepTaxon's "\[Discovery\]" output into something a deployment can trust.

**Judging the retrieval decision directly.** Moskvoretskii et al. showed that QA accuracy is a poor proxy for whether adaptive methods know when they need help ([Moskvoretskii et al.](https://arxiv.org/html/2501.12835v1)). Learned routers fall from 95.81% to 71.29% accuracy out of domain ([UniversalRAG](https://arxiv.org/pdf/2504.20734)). A meta-evaluation of multimodal gates would test decision-quality metrics and out-of-distribution shifts in question type. Shifts in image domain, as from lab to field, are what niche deployments will face.

**DINOv3 patch MaxSim inside a real QA loop.** The core experiment nobody has run is DINOv3 as retriever in a full image-QA RAG pipeline with QA accuracy reported. Compare four retrieval setups: global-token only, global plus patch MaxSim, MaxSim restricted to the grounded region, and SigLIP 2 alone. Test at several merge and projection settings on two or three niche domains.

**Hybrid index routing by query type.** No published evaluation tests a DINOv3 + SigLIP hybrid index for multimodal RAG. A small learned router could decide per query between image-to-image, image-to-text and fused search, learning from which direction retrieved the evidence. The E-VQA/InfoSeek split is the motivating evidence ([ReflectiVA](https://arxiv.org/abs/2411.16863)).

**Do the gains stack?** Reranker gains (+2 to +14), relevance-filter gains (\~9–12) and subject-crop gains have never been measured together under one retriever and generator ([Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720); [ReflectiVA](https://arxiv.org/abs/2411.16863)). An additive-ablation study would tell practitioners which component to build first.

**Quantizing image embeddings.** No retention numbers exist for int8, binary or RaBitQ codes of DINOv3 or SigLIP global and patch vectors, nor for MUVERA or merging applied to natural-image patches ([Qdrant quantization](https://qdrant.tech/documentation/manage-data/quantization/); [Light-ColPali](https://arxiv.org/html/2506.04997v1)). This is a contained, publishable systems study that bears directly on cost.

**Diffusion counterfactual hard negatives for niche retrievers.** TripletCLIP-style and CounterCurate-style gains were measured on compositional benchmarks, not fine-grained niche classes ([TripletCLIP](https://arxiv.org/abs/2411.02545); [CounterCurate](https://arxiv.org/abs/2402.13254)). The experiment: generate trait-flipped pairs ("same specimen, different spot pattern"), filter them with experts, contrastively train a projection over frozen DINOv3, and compare against a retrieval-only baseline. Geng et al. show that baseline is the one that matters ([Geng et al.](https://arxiv.org/html/2406.05184v2)).

**Restore-then-retrieve and domain imagine-then-retrieve.** No study quantifies whether super-resolution of low-quality queries improves or corrupts fine-grained nearest-neighbor retrieval. Given unchanged hallucination rates ([MedVAE SR](https://arxiv.org/abs/2604.12152)), the effect could go either way. Likewise, no study pairs generated proxies with a domain-tuned DINO encoder on fine-grained retrieval. The prediction is that the gain depends on whether the generator can draw the concept, which is itself measurable.

**Gallery curation economics.** Nobody has quantified the minimum gallery size per class at which RAG beats a fine-tuned small classifier, or the value of expert-in-the-loop gallery growth. An active-learning study would answer both: pick which production queries to send to experts by retrieval uncertainty, and plot accuracy against expert-hours.

## Sources

Pages opened during the research, grouped by topic. Several medical, species and drawing results are self-reported preprints without independent replication.

### Architectures, embedders, rerankers and KB-VQA

- [ColPali](https://arxiv.org/abs/2407.01449)
- [VisRAG](https://arxiv.org/abs/2410.10594)
- [MMDocIR](https://arxiv.org/abs/2501.08828)
- [ViDoRe V3](https://arxiv.org/abs/2601.08620)
- [Doc-RAG survey ("Scaling Beyond Context")](https://arxiv.org/abs/2510.15253)
- [Multimodal-RAG-Survey repository](https://github.com/llm-lab-org/Multimodal-RAG-Survey)
- [EchoSight](https://arxiv.org/abs/2407.12735)
- [InfoSeek](https://arxiv.org/abs/2302.11713)
- [ReflectiVA](https://arxiv.org/abs/2411.16863) ([HTML](https://arxiv.org/html/2411.16863v2))
- [MRAG-Bench](https://arxiv.org/abs/2410.08182) ([ICLR 2025 listing](https://mlanthology.org/iclr/2025/hu2025iclr-mragbench))
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [jina-embeddings-v4](https://arxiv.org/abs/2506.18902)
- [VLM2Vec](https://arxiv.org/abs/2410.05160) ([VLM2Vec-V2](https://arxiv.org/abs/2507.04590))
- [GME](https://arxiv.org/abs/2412.16855)
- [MM-Embed](https://arxiv.org/abs/2411.02571)
- [Qwen3-VL-Embedding and Reranker](https://arxiv.org/abs/2601.04720)
- [LamRA](https://arxiv.org/abs/2412.01720)
- [M3DocRAG](https://arxiv.org/abs/2411.04952)
- [VDocRAG](https://arxiv.org/abs/2504.09795)
- [RA-VQA](https://arxiv.org/abs/2210.03809)
- [FLMR](https://arxiv.org/abs/2309.17133)
- [PreFLMR](https://arxiv.org/abs/2402.08327)
- [Wiki-LLaVA](https://arxiv.org/abs/2404.15406)
- [RoRA-VLM](https://arxiv.org/abs/2410.08876)
- [mR²AG](https://arxiv.org/abs/2411.15041) ([HTML](https://arxiv.org/html/2411.15041v1))
- [OMGM](https://arxiv.org/abs/2505.07879)
- [ReAG (CVPR 2026)](https://arxiv.org/abs/2511.22715) ([supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Compagnoni_ReAG_Reasoning-Augmented_Generation_CVPR_2026_supplemental.pdf))
- [jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0)
- [RagVL](https://arxiv.org/abs/2407.21439)
- [mKG-RAG](https://arxiv.org/abs/2508.05318)
- [MR-MKG](https://arxiv.org/abs/2406.02030)
- [Visual Haystacks](https://arxiv.org/abs/2407.13766)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [Visual-RAG](https://arxiv.org/abs/2502.16636) ([overview](https://alphaxiv.org/overview/2502.16636v2))
- [Encyclopedic-VQA](https://arxiv.org/abs/2306.09224)
- [Dyn-VQA / OmniSearch](https://arxiv.org/abs/2411.02937) ([HTML](https://arxiv.org/html/2411.02937v5))
- [M2RAG / MM-RAIT](https://arxiv.org/abs/2502.17297)

### Domain case studies, failure modes and evaluation

- [RULE](https://arxiv.org/html/2407.05131v2)
- [MMed-RAG](https://www.arxiv.org/pdf/2410.13085)
- [FactMM-RAG](https://aclanthology.org/2025.naacl-long.28)
- [Path-RAG](https://arxiv.org/pdf/2411.17073)
- [Patho-AgenticRAG (overview)](https://www.alphaxiv.org/overview/2508.02258)
- [RS-RAG](https://arxiv.org/html/2504.04988v2)
- [ImageRAG for ultra-high-resolution remote sensing](https://arxiv.org/abs/2411.07688)
- [Snap'n Diagnose](https://arxiv.org/html/2408.14723v1)
- [AgroBench](https://openaccess.thecvf.com/content/ICCV2025/papers/Shinoda_AgroBench_Vision-Language_Model_Benchmark_in_Agriculture_ICCV_2025_paper.pdf)
- [VR-RAG](https://arxiv.org/html/2505.05635v2)
- [DeepTaxon](https://arxiv.org/html/2604.24029)
- [ReMP-AD](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.pdf)
- [Echo (industrial anomaly detection)](https://www.emergentmind.com/papers/2501.15795)
- [ArtSeek](https://www.emergentmind.com/papers/2507.21917)
- [Blueprint](https://arxiv.org/html/2602.13345)
- [MechVQA](https://huggingface.co/papers/2605.30794)
- [RAG-Driver](https://arxiv.org/html/2402.10828v2)
- [RAR](https://liuziyu77.github.io/RAR)
- [Baldassini et al., multimodal ICL](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Baldassini_What_Makes_Multimodal_In-Context_Learning_Work_CVPRW_2024_paper.pdf)
- [Who is in the Spotlight (position bias)](https://www.emergentmind.com/papers/2506.11063)
- [RAG-Check](https://huggingface.co/papers/2501.03995)
- [MLLM-as-a-Judge](https://alphaxiv.org/overview/2402.04788v3)
- [SWAT](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Few-Shot_Recognition_via_Stage-Wise_Retrieval-Augmented_Finetuning_CVPR_2025_paper.html)
- [SEM-VLM](https://pubs.rsc.org/en/content/articlehtml/2025/nr/d5nr03027g)

### DINOv3 and self-supervised encoders

- [DINOv3 paper](https://arxiv.org/abs/2508.10104) ([HTML](https://arxiv.org/html/2508.10104v1), [TMLR listing](https://mlanthology.org/tmlr/2026/simeoni2026tmlr-dinov3/))
- [Meta AI blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)
- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3) ([LICENSE](https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md))
- [NVIDIA TAO DINOv3](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/self_supervised_learning/dinov3.html)
- [SegRAG](https://www.alphaxiv.org/abs/2605.17630.md)
- [Memory-SAM](https://arxiv.org/abs/2510.15849)
- [FSSDINO](https://arxiv.org/abs/2602.07550)
- [DuoAD](https://arxiv.org/abs/2607.23924)
- [dino.txt (DINOv2 Meets Text)](https://arxiv.org/abs/2412.16334)
- [Talk2DINO](https://arxiv.org/abs/2411.19331)
- [DINOde](https://arxiv.org/abs/2607.21371)
- [Granulon](https://arxiv.org/abs/2603.08800)
- [CoME-VL](https://arxiv.org/abs/2604.03231)
- [What Makes CLIP Better than DINO?](https://arxiv.org/abs/2510.11835)
- [Scaling Language-Free Visual Representation Learning](https://arxiv.org/abs/2504.01017)
- [Does DINOv3 Set a New Medical Vision Standard?](https://arxiv.org/abs/2509.06467)
- [DINOv3 LoRA for MIDOG 2025](https://arxiv.org/abs/2508.21041)
- [FoundAD](https://arxiv.org/abs/2510.01934)
- [DC-VLAQ](https://arxiv.org/abs/2601.12729)

### Diffusion models

- [DIFT](https://arxiv.org/abs/2306.03881)
- [Diffusion Hyperfeatures](https://arxiv.org/abs/2305.14334)
- [A Tale of Two Features (SD + DINO)](https://arxiv.org/html/2305.15347v2)
- [CleanDIFT](https://arxiv.org/abs/2412.03439)
- [Diffusion Classifier](https://arxiv.org/abs/2303.16203) ([ar5iv](https://ar5iv.labs.arxiv.org/html/2303.16203))
- [REPA](https://arxiv.org/abs/2410.06940)
- [Diffusion sketch-photo retrieval](https://arxiv.org/abs/2403.07214v1)
- [DIVA](https://arxiv.org/abs/2407.20171)
- [The Unmet Promise of Synthetic Training Images (Geng et al.)](https://arxiv.org/html/2406.05184v2)
- [DA-Fusion](https://arxiv.org/abs/2302.07944)
- [Diff-Mix](https://arxiv.org/abs/2403.19600)
- [TripletCLIP](https://arxiv.org/abs/2411.02545)
- [CounterCurate](https://arxiv.org/abs/2402.13254)
- [RoentGen](https://arxiv.org/abs/2211.12737)
- [Ktena et al.](https://arxiv.org/abs/2304.09218)
- [DiffusionSat](https://arxiv.org/abs/2312.03606)
- [AnomalyDiffusion](https://arxiv.org/abs/2312.05767)
- [Self-Consuming Generative Models Go MAD](https://arxiv.org/abs/2307.01850)
- [GenIR](https://papers.neurips.cc/paper_files/paper/2025/file/c87da7d6ba0fb37236df181db9c2b8ff-Paper-Conference.pdf)
- [DAR](https://arxiv.org/abs/2501.15379)
- [DMCL](https://arxiv.org/pdf/2601.20391)
- [IP-CIR](https://arxiv.org/abs/2411.16752)
- [Category-level text-to-image retrieval with diffusion](https://arxiv.org/abs/2509.00177v1)
- [RDM](https://arxiv.org/abs/2204.11824)
- [Re-Imagen](https://arxiv.org/abs/2209.14491)
- [ImageRAG (retrieval-augmented generation)](https://arxiv.org/abs/2502.09411)
- [ACE](https://arxiv.org/abs/2303.09962)
- [Atad et al., diffusion-autoencoder counterfactuals](https://arxiv.org/abs/2408.01571)
- [Domain-specific latents for medical super-resolution](https://arxiv.org/abs/2604.12152)
- [RPU](https://www.alphaxiv.org/abs/2606.02331.md)
- [Janus](https://arxiv.org/abs/2410.13848)
- [BLIP3-o](https://arxiv.org/abs/2505.09568)
- [Transfusion](https://arxiv.org/abs/2408.11039)
- [OmniBridge](https://arxiv.org/html/2509.19018v1)
- [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042)
- [Carlini et al., extracting training data](https://arxiv.org/abs/2301.13188)
- [Somepalli et al., diffusion replication](https://arxiv.org/abs/2212.03860)

### Adaptive and agentic retrieval

- [Adaptive-RAG](https://arxiv.org/abs/2403.14403) ([PDF v2](https://arxiv.org/pdf/2403.14403v2))
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [CRAG](https://arxiv.org/abs/2401.15884)
- [FLARE](https://arxiv.org/abs/2305.06983)
- [DRAGIN](https://arxiv.org/abs/2403.10081)
- [SKR](https://arxiv.org/abs/2310.05002)
- [Probing-RAG](https://arxiv.org/html/2410.13339v2)
- [Adaptive Retrieval without Self-Knowledge? (Moskvoretskii et al.)](https://arxiv.org/html/2501.12835v1)
- [SURf](https://arxiv.org/abs/2409.14083)
- [MMSearch-R1](https://arxiv.org/pdf/2506.20670) ([ACL 2026](https://aclanthology.org/2026.acl-long.114.pdf))
- [DeepMMSearch-R1](https://arxiv.org/html/2510.12801v1)
- [Visual-ARFT](https://arxiv.org/abs/2505.14246)
- [VRAG-RL](https://arxiv.org/html/2505.22019v2)
- [WebWatcher](https://arxiv.org/abs/2508.05748)
- [VSearcher](https://arxiv.org/html/2603.02795)
- [PMSR](https://aclanthology.org/2026.acl-long.881.pdf)
- [UniversalRAG](https://arxiv.org/pdf/2504.20734)
- [MMA-RAG](https://arxiv.org/pdf/2603.00511)
- [DeepEyes](https://arxiv.org/pdf/2505.14362v3)
- [ZoomEye](https://arxiv.org/abs/2411.16044)
- [sCoT / SubagentVL](https://arxiv.org/pdf/2512.08511v2)
- [Detecting Knowledge Boundary of VLLMs](https://arxiv.org/html/2502.18023v3)
- [Do LVLMs Know What They Know?](https://arxiv.org/pdf/2508.19111)

### Vector indexes, quantization and databases

- [FAISS index guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [DiskANN](https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
- [FAISS + cuVS (Meta Engineering)](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/)
- [Milvus GPU CAGRA](https://milvus.io/blog/faster-index-builds-and-scalable-queries-with-gpu-cagra-in-milvus.md)
- [Weaviate HFresh](https://weaviate.io/blog/hfresh)
- [turbopuffer architecture](https://turbopuffer.com/docs/architecture) ([ANN v3](https://turbopuffer.com/blog/ann-v3))
- [Hugging Face embedding quantization](https://huggingface.co/blog/embedding-quantization)
- [Qdrant quantization](https://qdrant.tech/documentation/manage-data/quantization/)
- [Qdrant 1.16 release](https://qdrant.tech/blog/qdrant-1.16.x/index.md)
- [RaBitQ](https://arxiv.org/abs/2405.12497v1)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [Vespa binarizing vectors](https://docs.vespa.ai/en/rag/binarizing-vectors.html)
- [pgvectorscale reference](https://www.tigerdata.com/docs/reference/pgvectorscale)
- [pgvector 0.8.0](https://www.postgresql.org/about/news/pgvector-0.8.0-released-2952/)
- [PLAID](https://arxiv.org/abs/2205.09707v1)
- [MUVERA](https://arxiv.org/html/2405.19504v2)
- [Weaviate MUVERA](https://weaviate.io/blog/muvera) ([multi-vector docs](https://docs.weaviate.io/weaviate/configuration/compression/multi-vectors))
- [Token pooling](https://arxiv.org/html/2409.14683v1)
- [Light-ColPali](https://arxiv.org/html/2506.04997v1)
- [Elastic late-interaction at scale](https://www.elastic.co/search-labs/blog/late-interaction-model-colpali-scale)
- [Qdrant multivector tutorial](https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/)
- [Qdrant grouping](https://qdrant.tech/course/beginners/module-6/grouping/)
- [Milvus Array of Structs](https://milvus.io/blog/unlocking-true-entity-level-retrieval-new-array-of-structs-and-max-sim-capabilities-in-milvus.md) ([embedding lists docs](https://milvus.io/docs/search-with-embedding-lists.md))
- [LanceDB multivector](https://docs.lancedb.com/search/multivector-search)
- [Qdrant filtered search and ACORN](https://qdrant.tech/articles/filtered-vector-search-acorn/)
- [ACORN](https://arxiv.org/abs/2403.04871)
- [Big-ANN NeurIPS'23](https://arxiv.org/html/2409.17424v1)
- [Pinecone dedicated read nodes](https://docs.pinecone.io/guides/index-data/dedicated-read-nodes)
- [Chroma index reference](https://docs.trychroma.com/cloud/schema/index-reference.md)
- [VDBBench 1.0](https://zilliz.com/blog/vdbbench-1-0-benchmarking-with-your-real-world-production-workloads)
