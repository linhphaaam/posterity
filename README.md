# Posterity

**Posterity** is an interactive visualization system designed to explore a curated collection of American labor posters (1900–2010). The system combines temporal, thematic, and similarity-based views to support critical engagement with patterns of representation, omission, and symbolic continuity in visual archives.

This repository contains code for an initial prototype. A full release—including system components and dataset details—is forthcoming.

## Features

Posterity supports three core views:

- **Timeline View**: Visualizes poster activity over time, with overlays for themes, subject representation, and historical events.
- **3D Cloud View**: Clusters posters in a 3D space using PCA on CLIP-based multimodal embeddings derived from both image and metadata.
- **Similarity Spiral**: Allows users to submit images or gestures and explore semantically similar posters arranged in a 2D or 3D spiral layout.

## Repository Status

🔧 This repository contains partial code from early prototypes developed in p5.js and Python. It is intended as a reference snapshot of ongoing work.

- ✅ Initial layout and interaction sketches
- ✅ Data exploration scripts (Pandas, Matplotlib, Seaborn)
- ⚠️ Core embedding and spiral view logic to be released in a future update

## Dataset Access

The dataset of digitized American labor posters is currently under restricted access in collaboration with the Harvard Radcliffe Schlesinger Library. We are working with the archivist to determine appropriate pathways for sharin
