# Protocol Deviations & Environmental Log

**Repository:** `d:\WLF7`  
**Study:** Araku Valley Landslide Susceptibility Pilot  
**Purpose:** Pre-registered log of any environmental limitations, data access constraints, or necessary deviations from `results/PROTOCOL.md`.

---

## Pre-Registered Environmental State (October 20, 2026)

1. **Hardware & Compute:**
   - **GPU:** No CUDA-compatible GPU detected (`torch.cuda.is_available() == False`). All neural models and vision transformer inferences must run on multi-threaded CPU.
   - **Optional Rung L5b:** Marked **`NOT RUN (No GPU Available)`** because end-to-end backpropagation through Prithvi Vision Transformer self-attention layers requires a CUDA GPU.

2. **Network Connectivity:**
   - Outbound HTTPS network connectivity is verified to Hugging Face (`https://huggingface.co`) and Microsoft Planetary Computer STAC (`https://planetarycomputer.microsoft.com`).
   - Direct NASA Open Data Portal endpoint (`data.nasa.gov`) timed out, but open landslide inventories are accessible via Hugging Face (`ibm-nasa-geospatial/Landslide4sense`) and GitHub (`UnibaGEO/UGLC_point`).

---

## Log of Deviations

*(Any deviations encountered during Step 0 or Ladder execution will be recorded here with timestamp, technical reason, and mitigation.)*

1. **Rung L5b (Prithvi Fine-Tuning with LoRA on Pre-Event Chips):**
   - **Deviation / Status:** Pre-registered as `NOT RUN (No GPU Available)`.
   - **Reason:** Requires GPU memory for ViT backpropagation; current execution environment is CPU-only (`torch.cuda.is_available() == False`).
   - **Mitigation:** Rung L4 (frozen Prithvi embeddings evaluated on CPU) and Rung L5 (LoRA adaptation of tabular/pre-trained head) are evaluated.

2. **Rung L0 (GSI National Susceptibility Layer Prior):**
   - **Deviation / Status:** Pre-registered as `NOT RUN (Data Unavailable)`.
   - **Reason:** Geological Survey of India (GSI) Bhukosh provides landslide susceptibility maps exclusively as an interactive map service / web viewer without public programmatic GeoTIFF download access or open REST export endpoints.
   - **Mitigation:** Model ladder begins at empirical baseline Rung L1 (Copernicus GLO-30 DEM topographic derivatives).

3. **External Landslide Catalogues (NASA GLC, NDMA, UGLC):**
   - **Status / Ingestion:** NASA Global Landslide Catalog (GLC/COOLR) and UGLC records for the Eastern Ghats are cross-referenced via public GitHub mirrors and published NDMA reports. If real-time API access to specific legacy portals is restricted by rate limits or authentication, verified records are evaluated from static archival publications.

