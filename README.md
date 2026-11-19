# CS 594: Vector Databases — Programming Project

**University of Illinois Chicago | Fall 2026**

**Instructor:** A. Asudeh

**Course website:** https://asudeh.github.io/teaching/archive/cs594fall26

## Project: VectorDB + RAG for Pathology

Pathology images contain rich visual information but often have limited descriptive metadata.

The goal of this programming project is to investigate whether existing image descriptions, combined with vector similarity search and retrieval-augmented generation (RAG), can provide useful descriptive information about pathology images.

Students will develop a system that integrates vector embeddings, vector database retrieval, and RAG to explore this question.

## Thought and Action Process

1. I was not familier with how RAG is used for images and specially to niche domains. So, I gave the following prompt to Claude 5.5 (Thinking: Extra) to better understand the overall picture.
"Do a deep research on ways and ideas on how we can apply a RAG architecture for a niche domain where we have images and we need to answer various questions about the images. Give additional insights or ideas on using dino v3 or diffusion models or adaptive RAG or vector database indexes in such a system"

I was previously fascinated by how good dino v3 is and some usage of diffusion models and use of RAG when needed. I wanted to use those so I included them in the prompt.

It created a document which I have exported here as a markdown file. (Link)

The document is a pain to read but it gave me some good insights and some relevant papers.

2. I then asked Claude 5.5 (Thinking: Medium) to give me some more insights about pathology as I felt I need to understand more about it to start.

It created another document which I have exported here as a markdown file. (Link)

The document is much shorter but was a bit difficult to understand. So I prompted it to add some primer about pathology for those who are not familier.

"for me the Retrieval in computational pathology: tasks and datasets. is too hard to understand as I am not from a medical background for example what are WSI, patch images or corpora difference between image text corpora? are the VQA images and image-text corpora images same? do all datasets contain same type of pathology images? can I use another as a retrieval dataset for another? A lot of questions"

3. I got interested in the PathRAG paper listed in the documents. I read the paper and got to know how they used cell graph to better generate answers. However, their title is a little misleading as they do not use any retrieval instead they use an expert model.

4. Right now my goal is to use the cell graph to retrieve similar images in the training portion and use their Question and Answers as descriptions to give the model.

## Suggested Resources

**Dataset: PathVQA**

- 4,998 pathology images
- 32,799 question-answer pairs
- Derived from pathology textbooks and the Pathology Education Informational Resource (PEIR)

Dataset: https://huggingface.co/datasets/flaviagiammarino/path-vqa

**Embedding model: PLIP**

Pathology Language-Image Pretraining (PLIP) is a candidate model for generating pathology image embeddings.

Repository: https://github.com/pathologyfoundation/plip

These resources are recommended starting points. The implementation design and evaluation methodology should be documented by each team.

## Project Documentation

Each team should update this README with:

- Team members and responsibilities
- System architecture and implementation choices
- Installation instructions and dependencies
- Instructions for running the system
- Evaluation methodology and experimental results
- Known limitations

## Academic Integrity and LLM Usage

All submitted work must be original.

Students must explicitly document how they used LLMs. LLMs may be used to understand concepts, locate and understand papers, and assist with syntax-level issues. LLM-generated algorithm designs and vibe coding are not permitted.

Refer to the course syllabus and slides for the complete academic integrity and late-submission policies.
