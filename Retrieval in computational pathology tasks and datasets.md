# Retrieval in computational pathology: tasks and datasets

Nov 18, 2026 · @Shihab

## Primer for non-pathologists

Read this first. Pathology data comes at very different zoom levels and from very different places, and that decides which datasets can be mixed.

### What the image words mean

A pathologist cuts a thin slice of tissue (a biopsy or surgical specimen), puts it on a glass slide, stains it so cells become visible, and looks at it under a microscope.

&#91;embedded content: zoom levels · slide, WSI, ROI, patch\]

Each step zooms in. Datasets differ mainly in which of these levels they store.

| Term | Plain meaning | Rough size | Analogy |
| --- | --- | --- | --- |
| **Whole-slide image (WSI)** | The entire glass slide scanned at microscope resolution | Up to about 120,000 × 120,000 pixels, often several GB per file ([Quilt](https://pmc.ncbi.nlm.nih.gov/articles/PMC11090501); [SISH](https://link.springer.com/10.1038/s41551-022-00929-8)) | A full satellite map of a country |
| **Region of interest (ROI)** | A crop a pathologist marked as important | Thousands of pixels per side, e.g. 2048 × 1536 in BACH ([dataset list](https://github.com/maduc7/Histopathology-Datasets)) | One city on that map |
| **Patch or tile** | A small square cut from a WSI, the unit most models actually read | 96 to 1000 pixels per side, often 224 × 224 | One street block |
| **Mosaic** | A few hundred patches picked to represent a whole WSI, used by search engines | \~100 patches per slide | A photo album of the country's most typical blocks |
| **Magnification (20×, 40×)** | How zoomed in the scan is; at 20×, one pixel is about 0.5 µm | — | Map scale |
| **H&E stain** | The default stain: nuclei purple, tissue pink | — | The standard map style |
| **IHC and special stains** | Stains that light up one protein or structure, often brown | — | A themed map (only roads, only rivers) |
| **Label / diagnosis / subtype** | What the tissue is, e.g. "lung adenocarcinoma"; a subtype is a finer category of a cancer | — | The place name |
| **Pathology report** | The written diagnosis for a case, one per patient or slide | A few paragraphs | The map's legend and notes |

Why patches exist: a WSI is far too large to feed a model, so models read patches. A slide-level answer is then built by combining patch results.

### Five kinds of dataset

1. **Slide archives** (TCGA, CAMELYON, BRACS): real clinical WSIs from hospital scanners, each with a diagnosis label and sometimes a report.
2. **Patch datasets** (NCT-CRC-HE-100K, Kimia Path24, BreakHis): patches pre-cut from WSIs, each with a tissue label. They are small, clean and easy to use, but cover one organ or a few classes.
3. **Image-text corpora** (ARCH, OpenPath, Quilt-1M, CONCH data, PathGen-1.6M): pathology pictures paired with a sentence or caption. "Corpus" just means a large collection; "corpora" is the plural. They are mainly training data that teaches a model to connect pictures and words, but they can also serve as a searchable knowledge base.
4. **VQA benchmarks** (PathVQA, PathMMU, Quilt-VQA): images paired with questions and answers. They are test sets for question answering, not knowledge sources.
5. **Textbooks and reports used as RAG knowledge** (e.g. textbook pages in Patho-AgenticRAG): text or page images the model reads at answer time.

The difference between an image-text corpus and a VQA benchmark is the target. A corpus item says "this picture shows X"; a VQA item asks "what does this picture show?" and keeps the answer hidden for scoring.

### Are the images the same across datasets?

No. They differ in four ways:

- **Zoom unit.** Slide archives hold whole slides; patch datasets hold small tiles; image-text corpora and VQA sets mostly hold figure-sized pictures (a textbook photo, a video frame, a tweet image), which are closer to ROIs.
- **Capture.** Clinical archives come from slide scanners. Quilt images are frames of narrated YouTube videos, OpenPath images are Twitter posts, and PathVQA images come from textbooks and an online library. Video frames and tweets can be blurry, cropped or annotated with arrows.
- **Stain and organ.** CONCH data mixes H&E with IHC and special stains; most patch datasets are H&E only and cover one organ.
- **Labels.** Slide archives and patch sets carry class labels; corpora carry free text; VQA sets carry question-answer pairs.

**They do share sources, though, which matters for fairness.** Quilt-VQA was made from the same YouTube videos as Quilt-1M ([summary](https://arxiv.org/pdf/2608.03508)). PathMMU draws questions from PubMed, YouTube educational content (via Quilt), social media and atlases ([PathMMU](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07899.pdf)), which overlap with Quilt-1M, OpenPath and PubMed-based corpora such as CONCH's. PathGen-1.6M is built from TCGA slides ([PathGen](https://ar5iv.labs.arxiv.org/html/2407.00203)).

### Can one dataset be the retrieval database for another?

Yes, if three conditions hold:

1. **Same unit and similar look.** A patch query should search a patch database at a similar magnification and stain. Searching textbook photos with a scanner patch works less well than searching other scanner patches.
2. **The database holds what the answer needs.** For "what is this?" questions, a database of labeled images or captions works. For "what should be done?" questions you need text, such as textbook pages or reports.
3. **No leakage.** The test images, or near-copies of them, must not be in the database. Otherwise the system finds the answer instead of reasoning to it.

| Retrieval database | Evaluate on | Verdict |
| --- | --- | --- |
| Quilt-1M | Quilt-VQA | Risky: same source videos, likely near-duplicate frames |
| Quilt-1M or OpenPath | PathMMU | Risky for its EduContent and social-media subsets; filter duplicates first |
| CONCH-style PubMed captions | PathMMU PubMed subset | Risky: both drawn from PubMed figures |
| Textbook pages | PathVQA | Check carefully: PathVQA itself was built from two textbooks |
| ARCH or OpenPath captions | PathVQA, with duplicates removed | Reasonable for cross-modal "what is this" questions |
| TCGA slides | CPTAC or in-house slides | Good: SISH did exactly this as cross-cohort retrieval ([SISH supplement](https://rcastoragev2.blob.core.windows.net/5d60537189b855398d4606eda2b9e451/41551_2022_929_MOESM1_ESM.pdf)) |
| TCGA slides | TCGA slides | Only with leave-one-patient-out, as in [Lei et al.](https://arxiv.org/html/2605.00902) |
| NCT-CRC-HE-100K patches | Another colorectal patch set | Good if magnification and stain match; colorectal only |

Also check the license: ARCH is non-commercial (CC BY-NC-SA 4.0) ([ARCH](https://warwick.ac.uk/fac/cross_fac/tia/data/arch)), and OpenPath images must be fetched through the Twitter API ([PathGen\_init](https://huggingface.co/datasets/jamessyx/PathGen_init/blob/main/README.md)).

Deduplication tip: before evaluating, embed every test image and every database image with the same encoder, and drop database items whose cosine similarity to a test image is above a high threshold (e.g. 0.95). Then spot-check a few pairs by eye.

## Overview

Pathology retrieval has three layers. The oldest is content-based image search, where a query patch or whole-slide image (WSI) fetches similar diagnosed cases. Next is cross-modal search, where text finds images or images find captions. The newest is retrieval-augmented generation (RAG), where a pathology VLM answers questions using retrieved textbook pages, reports or image regions.

The field's core retrieval problem is still unsolved at slide level. On 9,387 TCGA diagnostic slides covering 17 organs and 60 diagnoses, the best of four slide foundation models (TITAN) reached only about 0.69 macro-F1 under leave-one-patient-out retrieval. Several diagnoses failed for every model ([Lei et al., 2026](https://arxiv.org/html/2605.00902)).

Three facts shape every design choice:

- **Gigapixel inputs.** A WSI can be several gigabytes, so search engines first reduce a slide to a "mosaic" of representative patches and index those ([SISH](https://link.springer.com/10.1038/s41551-022-00929-8); [Yottixel](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064517)).
- **Patch features matter more than aggregation.** Retrieval quality is driven mainly by the patch encoder; slide-level aggregation adds little ([Lei et al., 2026](https://arxiv.org/html/2605.00902)).
- **Data hygiene is a known weak spot.** SISH and RetCCL train and test on TCGA diagnostic slides, and reviewers could not verify the train/test separation ([Tizhoosh & Pantanowitz, 2024](https://arxiv.org/pdf/2401.08699)).

## Retrieval tasks

Pathology retrieval tasks are defined by the query unit (patch, slide, text) and by what counts as a correct match (same tissue, same subtype, same site, matching caption).

| Task | Query → database | A hit means | Typical metrics | Example benchmarks |
| --- | --- | --- | --- | --- |
| Patch-to-patch search | Tissue patch → patch archive | Same tissue type or source slide | Patch-to-scan and whole-scan accuracy; precision@k | Kimia Path24, NCT-CRC-HE-100K (Kather100k) |
| WSI-to-WSI search (disease subtype) | Slide → slides from the same organ | Same diagnosis | Majority vote at top-k (MV@k), mMV@k, mAP@k, macro-F1 | TCGA, CPTAC, BWH in-house |
| Anatomic site retrieval | Slide → all slides | Same organ | MV@k, accuracy | TCGA (25 sites) |
| Rare-disease search | Slide → slides incl. rare subtypes | Same rare diagnosis | MV@k | TCGA/CPTAC rare subtypes (SISH) |
| Text-to-image retrieval | Caption or query text → image archive | Paired image | Recall@k | OpenPath (Twitter), ARCH, PathPedia, CONCH held-out sets |
| Image-to-text retrieval | Image → captions or reports | Paired caption/report | Recall@k | ARCH, PathSearch slide-report pairs |
| Slide-to-report and report-to-slide | WSI ↔ pathology report | Matching case | Top-k accuracy | PathSearch cohorts, TCGA reports |
| Retrieval-augmented VQA | Image + question → textbook pages, regions or cases | Answer is correct | Accuracy, recall on open answers | PathVQA, PathMMU, Quilt-VQA |

How the main metrics work:

- **Majority vote at top-k (MV@k).** Introduced with Yottixel. A search counts as correct only if at least 51% of the top k retrieved slides carry the query's label. It is the most conservative of the common metrics ([Tizhoosh & Pantanowitz, 2024](https://arxiv.org/pdf/2401.08699)).
- **mMV@k and mAP@k.** SISH reports majority vote and average precision at k, averaged over classes, so rare subtypes count as much as common ones ([SISH](https://link.springer.com/10.1038/s41551-022-00929-8)).
- **Kimia Path24 accuracies.** Patch-to-scan accuracy counts how many of the 1,325 test patches retrieve a patch from their own source slide. Whole-scan accuracy averages that per slide, and total accuracy multiplies the two ([Gram Barcodes](https://arxiv.org/pdf/2111.15519)).
- **Leave-one-patient-out.** The fairest slide protocol removes every slide of the query's patient from the database before searching ([Lei et al., 2026](https://arxiv.org/html/2605.00902)).

Worked example of MV@5: a lung adenocarcinoma query returns five slides labeled LUAD, LUAD, LUSC, LUAD, LUSC. Three of five (60%) match, so the search counts as correct. With LUAD, LUSC, LUSC, LUAD, LUSC it fails, even though the top hit was right.

## Datasets

Pathology retrieval draws on four kinds of data: slide archives for case search, labeled patch sets for patch search, image-text corpora for cross-modal search, and VQA benchmarks for retrieval-augmented answering. Sizes are as reported by each source.

### Slide archives (case and WSI search)

| Dataset | Size | Content and labels | Retrieval use |
| --- | --- | --- | --- |
| [TCGA](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064517) | Slides from almost 11,000 patients; 25 anatomic sites, 32 cancer subtypes | Diagnostic (FFPE) and frozen-section H&E WSIs, with reports and omics | The main WSI-search benchmark: subtype and site retrieval (Yottixel, SISH, RetCCL, slide FMs) |
| [TCGA retrieval cohort (Lei et al.)](https://arxiv.org/html/2605.00902) | 9,387 diagnostic WSIs; 17 organs, 60 organ-specific diagnoses (from 11,765 WSIs, 9,640 patients) | Diagnoses with ≥4 patients, organs with ≥2 diagnoses | Leave-one-patient-out slide retrieval of foundation models |
| [SISH evaluation set](https://www.cancer.fr/professionnels-de-sante/veille/nota-bene-cancer/bulletin-n-540/fast-and-scalable-search-of-whole-slide-images-via-self-supervised-deep-learning) | Over 22,000 patient cases, 56 disease subtypes | TCGA, CPTAC and Brigham and Women's in-house slides | Subtype, site, rare-disease and cross-cohort retrieval |
| [CAMELYON16](https://github.com/maduc7/Histopathology-Datasets) | 400 WSIs (270 train: 160 normal, 110 metastatic; 130 test) | Sentinel lymph node H&E, binary masks | Tumor vs. normal slide and region search |
| [CAMELYON17](https://arxiv.org/pdf/2310.19656) | 899 training WSIs, 100 patients, 5 hospitals | Lymph node resections | Cross-site (domain shift) retrieval |
| [BRACS](https://github.com/maduc7/Histopathology-Datasets) | 547 WSIs, 4,539 ROIs, 189 patients | Breast; 7 fine classes (6 lesion subtypes + normal), 3 coarse | Fine-grained subtype retrieval at ROI and slide level |
| [PathSearch training corpus](https://www.emergentmind.com/papers/2510.23224) | 6,926 slide-report pairs | WSIs paired with diagnostic reports | Slide-to-report and text-to-slide retrieval |

### Patch datasets (patch search)

| Dataset | Size | Content and labels | Retrieval use |
| --- | --- | --- | --- |
| [Kimia Path24](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/html/Babaie_Classification_and_Retrieval_CVPR_2017_paper.html) | 24 WSIs; 1,325 test patches of 1000×1000 px (0.5 × 0.5 mm); \~27,000–50,000 training patches | 24 tissue textures, one per slide | Built for classification and retrieval; best original baseline 41.80% (CNN) |
| [NCT-CRC-HE-100K (Kather100k)](https://arxiv.org/pdf/2311.05659) | 100,000 patches, 224×224 px | Colorectal, 9 tissue classes; test set CRC-VAL-HE-7K | Patch retrieval in SISH and PLIP evaluations |
| [BreakHis](https://arxiv.org/pdf/2206.00092) | 7,909 images, 82 patients | Breast benign vs. malignant, 8 tumor types, at 40×–400× | Multi-magnification patch search |
| [LC25000](https://arxiv.org/pdf/2206.00092) | 25,000 images, 5 classes of 5,000 | Lung and colon benign/cancer | Patch classification and retrieval |
| [PatchCamelyon (PCam)](https://arxiv.org/pdf/2310.19656) | 327,680 patches of 96×96 px | From CAMELYON16, tumor vs. normal | Binary patch search |

### Image-text corpora (cross-modal search and retriever training)

| Dataset | Size | Source | Retrieval use |
| --- | --- | --- | --- |
| [ARCH](https://arxiv.org/abs/2103.05121v1) | About 8K samples | Textbooks and PubMed articles; multi-image bags with one caption | Standard image↔text retrieval test set |
| [OpenPath](https://huggingface.co/spaces/vinid/webplip/raw/main/details.py) | 208,414 image-text pairs | Twitter posts and replies across 32 pathology hashtags, plus LAION | Trains PLIP; text-to-image and image-to-image search |
| [Quilt / Quilt-1M](https://quilt1m.github.io) | Quilt: 768,826 pairs, 419,780 images from 4,504 YouTube videos (1,087 h); Quilt-1M: 1M pairs | Narrated educational videos, plus Twitter, papers and the web | Trains QuiltNet; retrieval pretraining |
| [CONCH pretraining data](https://huggingface.co/MahmoodLab/CONCH/blob/refs%2Fpr%2F14/README.md) | 1.17M image-caption pairs | PMC-OA plus internal sources; H&E, IHC and special stains | Trains CONCH, a strong zero-shot cross-modal retriever |
| [PathGen-1.6M](https://ar5iv.labs.arxiv.org/html/2407.00203) | 1.6M generated image-caption pairs | Representative TCGA WSI patches captioned by collaborating agent models | Trains PathGen-CLIP |

### VQA benchmarks (retrieval-augmented answering)

| Dataset | Size | Question format | Retrieval use |
| --- | --- | --- | --- |
| [PathVQA](https://arxiv.org/pdf/2003.10286) | 32,799 QA pairs, 4,998 images | Half open-ended (what, where, how), half yes/no; from two textbooks and PEIR | Path-RAG and agentic RAG evaluations |
| [PathMMU](https://arxiv.org/html/2401.16355v3) | 33,428 multiple-choice questions, 24,067 images | Expert-validated, with explanations; PubMed, social media, Atlas, EduContent, classification subsets | Patho-AgenticRAG and agent evaluations |
| [Quilt-VQA](https://arxiv.org/pdf/2608.03508) | 1,283 QA pairs on 985 images (940 open, 343 closed) | Questions narrated in Quilt videos | Yes/no and open accuracy for RAG systems |

## Dataset details

### TCGA: the default slide-search archive

Yottixel's evaluation searched the largest public WSI repository, covering almost 11,000 patients, 25 anatomic sites and 32 cancer subtypes. It used both permanent diagnostic slides and frozen sections, and never trained on TCGA ([Kalra et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064517)). Majority-vote accuracy was high for distinctive subtypes, for example 99% for skin cutaneous melanoma and 100% for thymoma on permanent slides ([Kalra et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064517)).

The 2026 Mayo Clinic benchmark rebuilt TCGA as a cleaner retrieval testbed ([Lei et al., 2026](https://arxiv.org/html/2605.00902)):

- **Cohort:** 9,387 diagnostic WSIs, 17 organs, 60 organ-specific diagnoses, after dropping diagnoses with fewer than four patients and organs with fewer than two diagnoses.
- **Protocol:** leave-one-patient-out; top-1 and top-3 majority vote; macro-F1 at patient level, averaged over organs.
- **Models:** slide foundation models TITAN, PRISM, Madeleine and GigaPath; ABMIL on CONCH v1.5 patch features; Yottixel with CONCH v1.5 at 5–100% patch retention.

| Model | Top-1 macro-F1 | Top-3 macro-F1 |
| --- | --- | --- |
| TITAN | 0.689 | 0.689 |
| ABMIL (CONCH v1.5) | 0.667 | 0.687 |
| Yottixel + CONCH, 100% patches | 0.656 | 0.674 |
| Yottixel + CONCH, 20% patches | 0.654 | 0.668 |
| Madeleine | 0.645 | 0.647 |
| PRISM | 0.632 | 0.643 |
| GigaPath | 0.590 | 0.583 |

Source: [Lei et al., 2026](https://arxiv.org/html/2605.00902). Standard deviations across organs are about ±0.20, so organ and diagnosis matter more than model choice. Errors cluster within the same organ or lineage, mirroring disagreements between pathologists.

### SISH cohorts: TCGA, CPTAC and an in-house test set

SISH was evaluated on over 22,000 patient cases and 56 disease subtypes ([SISH summary](https://www.cancer.fr/professionnels-de-sante/veille/nota-bene-cancer/bulletin-n-540/fast-and-scalable-search-of-whole-slide-images-via-self-supervised-deep-learning)). Its tasks were disease-subtype retrieval within an anatomic site, anatomic-site retrieval, rare-disease retrieval, cross-cohort retrieval (database built on TCGA, queries from CPTAC and BWH), and patch retrieval on Kather100k ([SISH supplement](https://rcastoragev2.blob.core.windows.net/5d60537189b855398d4606eda2b9e451/41551_2022_929_MOESM1_ESM.pdf)). Per-subtype mMV@5 on TCGA included 99.29% for kidney clear cell carcinoma, 83.18% for ovarian serous carcinoma and 74.23% for stomach adenocarcinoma ([SISH supplement](https://rcastoragev2.blob.core.windows.net/5d60537189b855398d4606eda2b9e451/41551_2022_929_MOESM1_ESM.pdf)).

An independent validation by Lahr et al. used four internal datasets (1,269 patients) and three public ones (1,207 patients): over 200,000 patches from 38 classes across five primary sites ([Lahr et al.](https://sotaverified.org/papers/analysis-and-validation-of-image-search)). It ranked Yottixel first overall, BoVW third, SISH fourth and RetCCL last ([Tizhoosh & Pantanowitz, 2024](https://arxiv.org/pdf/2401.08699)).

### Kimia Path24: a patch-retrieval stress test

Kimia Path24 cuts 24 WSIs of different tissue textures into 1,325 test patches of 1000×1000 pixels. Training patches are generated by the user, from about 27,000 to over 50,000 with the preset parameters ([Babaie et al., 2017](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/html/Babaie_Classification_and_Retrieval_CVPR_2017_paper.html)). Its compound patch-and-scan accuracy is deliberately strict: the best original baseline reached 41.80% ([Babaie et al., 2017](https://arxiv.org/pdf/1705.07522v1.pdf)).

### Image-text corpora

- **OpenPath** combines 116,504 pairs from tweets and 59,869 pairs from top-liked replies across 32 subspecialty hashtags (2006–2022), plus PathLAION, for 208,414 pairs ([PLIP demo](https://huggingface.co/spaces/vinid/webplip/raw/main/details.py); [PLIP](https://huggingface.co/spaces/vinid/webplip/blame/refs%2Fpr%2F8/introduction.md)). PLIP's text-to-image retrieval was validated on the Twitter hold-out set, PathPedia (210 pairs), and PubMed and book collections ([PLIP notes](https://creatis-myriad.github.io/2023/09/28/PLIP.html)).
- **Quilt** was mined from 4,504 narrated videos spanning 1,087 hours: 438K unique images with 768K text pairs, covering 1.469M UMLS entity mentions (28.5K unique) at magnifications from 0–10× to 20–40× ([Quilt-1M](https://quilt1m.github.io)). Quilt-1M adds Twitter, papers and web data to reach 1M pairs ([Quilt-1M](https://pmc.ncbi.nlm.nih.gov/articles/PMC11090501)).
- **CONCH data** has 1.17M pairs: 457,373 H&E and 713,595 IHC and special-stain pairs ([CONCH figure](https://biomedcentral-translational-medicine.publicaciones.saludcastillayleon.es/articles/10.1186/s12967-025-06428-z/figures/4)). CONCH did not pretrain on TCGA, PAIP or GTEx, which lowers contamination risk on those benchmarks ([CONCH](https://pypi.org/project/conch-fork)).
- **PathGen-1.6M** used agent models to select representative WSI patches and generate and refine captions. Its PathGen\_init release reuses 400K Quilt-1M images and 200K PathCap images, while OpenPath images must be fetched through the Twitter API ([PathGen](https://ar5iv.labs.arxiv.org/html/2407.00203); [PathGen\_init](https://huggingface.co/datasets/jamessyx/PathGen_init/blob/main/README.md)).

### VQA benchmarks

- **PathVQA** has 1,670 images from two textbooks and 3,328 from the PEIR library, with 6.6 questions per image on average. 16,465 questions (50.2%) are open-ended ([PathVQA](https://arxiv.org/pdf/2003.10286)). The Hugging Face release lists 5,004 images, of which 4,289 are used by a QA pair ([HF card](https://huggingface.co:443/datasets/flaviagiammarino/path-vqa/blob/main/README.md)).
- **PathMMU** draws 32% of questions from PubMed, 20% from YouTube EduContent, 19% from social media, 18% from classification datasets and 11% from atlases. Seven pathologists checked the validation and test sets ([PathMMU](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07899.pdf); [ECCV 2024](https://eccv2024.ecva.net/virtual/2024/oral/878)).
- **Quilt-VQA** contains 1,283 QA pairs on 985 images drawn from question-answer exchanges narrated in Quilt videos ([dataset summary](https://arxiv.org/pdf/2608.03508)).

## Retrieval and RAG systems

Search engines index mosaics of patches. Foundation models supply the embeddings. RAG systems wrap retrieval around a pathology VLM.

| System | Type | How retrieval works | Reported evaluation |
| --- | --- | --- | --- |
| [Yottixel](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064517) | WSI search engine | Unsupervised mosaic of patches, DenseNet features, binary barcodes | Whole TCGA (diagnostic + frozen) without training on it; best MV@5 and search speed in an independent validation ([review](https://arxiv.org/pdf/2401.08699)) |
| [SISH](https://link.springer.com/10.1038/s41551-022-00929-8) | WSI search engine | Yottixel-style mosaic, self-supervised discrete codes, van Emde Boas tree for constant-time candidate lookup | 22,000+ cases, 56 subtypes; heavy tree storage and inconsistent results in later validation ([review](https://arxiv.org/pdf/2401.08699)) |
| [RetCCL](https://ar5iv.labs.arxiv.org/html/2306.17019) | WSI search engine | Clustering-guided contrastive feature extractor on mosaics | Ranked last for accuracy and efficiency in Lahr et al. ([review](https://arxiv.org/pdf/2401.08699)) |
| [PLIP](https://huggingface.co/spaces/vinid/webplip/blame/refs%2Fpr%2F8/introduction.md) | Vision-language model | CLIP fine-tuned on OpenPath; text-to-image and image-to-image search | Twitter, PathPedia, PubMed and book validation sets |
| [CONCH](https://www.github.com/mahmoodlab/CONCH) | Vision-language model | CoCa-style pretraining on 1.17M pairs; zero-shot image↔text retrieval | State of the art across 14 tasks, including both retrieval directions |
| [TITAN](https://arxiv.org/html/2605.00902) | Slide foundation model | Slide embedding from patch features, pretrained on 335,645 WSIs | Best slide-retrieval macro-F1 (0.689) on the TCGA cohort |
| [PathSearch](https://arxiv.org/html/2510.23224v1) | Multimodal WSI search | Attentive mosaics with binarized Hamming search for image-to-image; slide embeddings aligned with reports for text queries | Trained on 6,926 slide-report pairs; up to +10.9% top-1 on subtyping and +7.4% on grading over image-to-image baselines |
| [Path-RAG](https://arxiv.org/abs/2411.17073) | RAG for VQA | HistoCartography selects key patches (cell-graph knowledge) to feed the VLM | PathVQA-Open; state of the art before it reached only 38% recall ([Path-RAG](https://arxiv.org/html/2411.17073v1)) |
| [Patho-AgenticRAG](https://www.alphaxiv.org/overview/2508.02258) | Agentic multimodal RAG | Page-level embeddings of pathology textbooks; agent decomposes tasks and retrieves over several turns, trained with RL | Reported 75.8 on Quilt-VQA yes/no and 78.32 on PathMMU test ([aggregator listing](https://www.wizwand.com/paper/68fa7a761e209131cea83672)) |

**Caveat:** the Patho-AgenticRAG figures come from a results aggregator rather than the paper itself, and the numbers in this table come from different datasets and metrics. Read each row on its own; do not rank systems across rows.

What the evidence says for building a pathology retriever:

- **Start from a strong patch encoder.** In the TCGA benchmark, Yottixel's mosaic search on CONCH v1.5 features came within about 0.03 macro-F1 of the best slide foundation model, even at 20% patch retention ([Lei et al., 2026](https://arxiv.org/html/2605.00902)).
- **Binarize for storage.** A 1,024-dimension float vector is 4,096 bytes, so 100 patches per slide is about 400 KB. Binary barcodes cut that to about 12 KB, a 97% reduction ([Tizhoosh & Pantanowitz, 2024](https://arxiv.org/pdf/2401.08699)).
- **Report per organ and per diagnosis.** Organ-averaged scores hide subtypes that fail completely ([Lei et al., 2026](https://arxiv.org/html/2605.00902)).
- **Keep pretraining data out of the benchmark.** Prefer encoders that never saw the evaluation archive, as Yottixel did with TCGA and CONCH with TCGA, PAIP and GTEx ([review](https://arxiv.org/pdf/2401.08699); [CONCH](https://pypi.org/project/conch-fork)).

## Sources

- [Validation of Whole-Slide Foundation Models for Image Retrieval in TCGA Data (Lei et al., 2026)](https://arxiv.org/html/2605.00902)
- [On Image Search in Histopathology (Tizhoosh & Pantanowitz, 2024)](https://arxiv.org/pdf/2401.08699)
- [Yottixel TCGA search (Kalra et al., 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7064517)
- [SISH, Nature Biomedical Engineering 2022](https://link.springer.com/10.1038/s41551-022-00929-8) ([supplement](https://rcastoragev2.blob.core.windows.net/5d60537189b855398d4606eda2b9e451/41551_2022_929_MOESM1_ESM.pdf), [summary](https://www.cancer.fr/professionnels-de-sante/veille/nota-bene-cancer/bulletin-n-540/fast-and-scalable-search-of-whole-slide-images-via-self-supervised-deep-learning))
- [Histopathology Slide Indexing and Search: Are We There Yet?](https://ar5iv.labs.arxiv.org/html/2306.17019)
- [Analysis and Validation of Image Search Engines in Histopathology (Lahr et al.)](https://sotaverified.org/papers/analysis-and-validation-of-image-search)
- [PathSearch (2025)](https://arxiv.org/html/2510.23224v1) ([summary](https://www.emergentmind.com/papers/2510.23224))
- [Kimia Path24 (Babaie et al., 2017)](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/html/Babaie_Classification_and_Retrieval_CVPR_2017_paper.html) ([arXiv](https://arxiv.org/pdf/1705.07522v1.pdf), [metrics](https://arxiv.org/pdf/2111.15519))
- [Histopathology Datasets list (BRACS, CAMELYON16, BreakHis)](https://github.com/maduc7/Histopathology-Datasets)
- [Domain Generalization in Computational Pathology (CAMELYON17, PCam)](https://arxiv.org/pdf/2310.19656)
- [FHIST benchmark (BreakHis, LC25000)](https://arxiv.org/pdf/2206.00092)
- [NCT-CRC-HE-100K details](https://arxiv.org/pdf/2311.05659)
- [ARCH (Gamper & Rajpoot, CVPR 2021)](https://arxiv.org/abs/2103.05121v1)
- [PLIP / OpenPath](https://huggingface.co/spaces/vinid/webplip/raw/main/details.py) ([notes](https://creatis-myriad.github.io/2023/09/28/PLIP.html))
- [Quilt-1M](https://quilt1m.github.io) ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11090501))
- [CONCH model card](https://huggingface.co/MahmoodLab/CONCH/blob/refs%2Fpr%2F14/README.md) ([GitHub](https://www.github.com/mahmoodlab/CONCH), [data figure](https://biomedcentral-translational-medicine.publicaciones.saludcastillayleon.es/articles/10.1186/s12967-025-06428-z/figures/4))
- [PathGen-1.6M](https://ar5iv.labs.arxiv.org/html/2407.00203) ([PathGen\_init](https://huggingface.co/datasets/jamessyx/PathGen_init/blob/main/README.md))
- [PathVQA](https://arxiv.org/pdf/2003.10286) ([Hugging Face card](https://huggingface.co:443/datasets/flaviagiammarino/path-vqa/blob/main/README.md))
- [PathMMU (ECCV 2024)](https://arxiv.org/html/2401.16355v3) ([paper PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07899.pdf))
- [Quilt-VQA description](https://arxiv.org/pdf/2608.03508)
- [Path-RAG](https://arxiv.org/abs/2411.17073)
- [Patho-AgenticRAG](https://www.alphaxiv.org/overview/2508.02258) ([results listing](https://www.wizwand.com/paper/68fa7a761e209131cea83672))
