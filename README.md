# TALA Backend Research

## An Exploratory Framework for Astronaut Exercise Movement Analysis

TALA is an exploratory research prototype that integrates computer vision, OpenSim-based kinematic reconstruction, biomechanical and movement analysis, and NASA-literature-grounded retrieval-augmented generation (RAG).

The backend processes exercise video to extract human pose landmarks, uses those observations to inform a musculoskeletal model-based kinematic reconstruction, derives movement and gravity-related quantities, and provides contextual explanations grounded in NASA human-health and spaceflight literature.

The system is intended as a research and demonstration prototype. Its outputs are exploratory and are not intended for clinical diagnosis, medical decision-making, or evaluation of astronaut fitness.

---

## 1. Project Overview

TALA's backend is organized into four principal stages:

1. **Computer Vision:** Extracts human pose landmarks from exercise video using MediaPipe Pose Landmarker.
2. **OpenSim Kinematic Reconstruction:** Maps visual observations to a musculoskeletal model and performs inverse kinematics to estimate model-constrained joint kinematics.
3. **Biomechanical and Movement Analysis:** Analyzes reconstructed movement, identifies movement phases and repetitions, calculates whole-body center of mass (COM), and evaluates relative gravitational potential energy under different gravity constants.
4. **NASA Literature-Grounded RAG:** Retrieves relevant passages from NASA documents and generates contextual explanations using a language model.

The stages are integrated through a unified assessment pipeline that combines movement observations, reconstruction-quality information, gravity context, daily check-in information, and retrieved NASA literature.

### Research Objectives

The backend explores the feasibility of:

- Extracting usable pose landmarks from astronaut exercise footage.
- Using monocular visual observations to inform OpenSim-based kinematic reconstruction.
- Deriving movement-related quantities from reconstructed kinematics.
- Comparing gravitational potential energy for a fixed reconstructed movement trajectory under different gravity constants.
- Retrieving and generating NASA-literature-grounded contextual explanations.
- Integrating these components into a reproducible assessment workflow.

---

## 2. System Architecture

```text
                  Exercise Video
                        |
                        v
              Computer Vision
              MediaPipe Pose
                        |
                        v
                Pose Landmarks
                        |
                        v
          OpenSim Kinematic Reconstruction
              Model Scaling / IK
                        |
                        v
             Reconstructed Kinematics
                        |
             +----------+----------+
             |                     |
             v                     v
      Movement Analysis      Gravity Analysis
      Phases / Repetitions    COM / Potential Energy
             |                     |
             +----------+----------+
                        |
                        v
              Assessment Pipeline
          Movement / Quality / Check-in
                        |
                        v
                 NASA Literature RAG
             Retrieval + Gemini Generation
                        |
                        v
                 Unified TALA Output
```

### Component Responsibilities

| Component                | Responsibility                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Pose Estimation          | Extracts image-based body landmarks from exercise video.                                                           |
| Kinematic Reconstruction | Uses visual targets and an OpenSim musculoskeletal model to estimate joint kinematics through inverse kinematics.  |
| Movement Analysis        | Identifies movement phases and repetitions and summarizes movement characteristics.                                |
| Gravity Analysis         | Calculates COM-related gravitational potential energy for specified gravity constants.                             |
| Assessment               | Combines movement observations, reconstruction quality, gravity context, and available daily check-in information. |
| NASA RAG                 | Retrieves relevant NASA literature and generates contextual explanations.                                          |

The RAG component provides supporting context. It does not modify the numerical calculations or independently determine assessment flags.

---

## 3. Repository Structure

```text
tala-backend/
│
├── configs/
│   ├── kinematic_reconstruction.yaml
│   ├── phase3_3_pose_extraction.yaml
│   └── rag.yaml
│
├── data/
│   ├── knowledge_base/
│   │   ├── raw/
│   │   │   ├── HRP-Musculoskeletal-Fitness.pdf
│   │   │   ├── HRP-Sleep-Circadian.pdf
│   │   │   ├── NASA-STD-3001-Volume-1.pdf
│   │   │   ├── NASA-STD-3001-Volume-2.pdf
│   │   │   ├── OCHMO-HB-004.pdf
│   │   │   └── OCHMO-TB-031.pdf
│   │   ├── chroma_db/
│   │   └── source_manifest.json
│   │
│   └── raw/
│       └── squat_segment.mp4
│
├── docs/
│   ├── architecture.md
│   ├── limitations.md
│   ├── methodology.md
│   └── rag_methodology.md
│
├── models/
│   ├── opensim/
│   │   └── RajagopalLaiUhlrich2023.osim
│   └── pose_landmarker/
│       └── pose_landmarker_full.task
│
├── results/
│   ├── assessment/
│   ├── baselines/
│   ├── gravity_analysis/
│   ├── kinematic_reconstruction/
│   ├── movement_analysis/
│   ├── pose_estimation/
│   └── tala_output.json
│
├── scripts/
│   ├── build_knowledge_base.py
│   ├── examine_movement.py
│   ├── run_assessment_example.py
│   ├── run_gravity_analysis.py
│   ├── run_kinematic_reconstruction.py
│   ├── run_movement_analysis.py
│   ├── run_pose_estimation.py
│   ├── run_tala.py
│   ├── test_gemini_connection.py
│   ├── test_rag_generation.py
│   └── test_rag_retrieval.py
│
├── src/
│   └── tala/
│       ├── assessment/
│       ├── baselines/
│       ├── common/
│       ├── kinematic_reconstruction/
│       ├── movement_analysis/
│       ├── pose_estimation/
│       ├── rag/
│       └── pipeline.py
│
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

Generated files and local databases may not be present in a fresh clone. Refer to the reproduction instructions below.

---

## 4. Requirements

### Software

- Python 3.11
- Git
- Ollama
- Google Gemini API key
- OpenSim-compatible Python environment and dependencies declared by the project
- Dependencies declared in `pyproject.toml`

The project was developed and tested using Python 3.11.9.

### Python Environment

For Windows PowerShell:

```powershell
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd tala-backend

# Create a virtual environment
py -3.11 -m venv .venv

# Activate the environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install the project
pip install -e .
```

If the project declares test dependencies as an optional extra, install that extra according to the configuration in `pyproject.toml`.

---

## 5. Environment Configuration

### Google Gemini

TALA uses Google Gemini for natural-language generation from retrieved NASA literature.

Create a `.env` file in the repository root:

```env
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Keep this file private. Never commit API keys or credentials to version control.

The repository includes `.env.example` as a configuration reference.

### Ollama Embeddings

TALA uses Ollama to generate local text embeddings for the NASA knowledge base.

The configured embedding model is:

```text
qwen3-embedding:0.6b
```

Install Ollama and retrieve the model:

```bash
ollama pull qwen3-embedding:0.6b
```

Ensure the Ollama service is running when building or querying the knowledge base.

**Model responsibilities:**

| Model / Service                     | Responsibility                                                    |
| ----------------------------------- | ----------------------------------------------------------------- |
| Qwen3 Embedding 0.6B through Ollama | Converts NASA document chunks and queries into vector embeddings. |
| ChromaDB                            | Stores and retrieves embedded document chunks.                    |
| Google Gemini                       | Generates contextual answers using retrieved evidence.            |

The embedding and generation stages are separate. Ollama handles embeddings, while Gemini handles answer generation.

---

## 6. Required Models and Data

### OpenSim Musculoskeletal Model

```text
models/opensim/RajagopalLaiUhlrich2023.osim
```

This model is used for model-based kinematic reconstruction.

### MediaPipe Pose Landmarker

```text
models/pose_landmarker/pose_landmarker_full.task
```

This model is used to extract body landmarks from the input video.

### Exercise Video

```text
data/raw/squat_segment.mp4
```

The video serves as the input for the pose-estimation stage.

### NASA Knowledge-Base Documents

The knowledge base uses six NASA-related source documents:

1. NASA-STD-3001 Volume 1 — Crew Health.
2. NASA-STD-3001 Volume 2.
3. OCHMO-HB-004 — Anthropometry, Biomechanics, and Strength.
4. OCHMO-TB-031 — Exercise Overview: Astronaut Physiological Deconditioning and Exercise Prescription Countermeasures in Spaceflight.
5. HRP Evidence Report — Reduced Cardiorespiratory and Musculoskeletal Fitness.
6. HRP Evidence Report — Sleep Loss, Circadian Desynchronization, and Work Overload.

The original PDFs are stored under:

```text
data/knowledge_base/raw/
```

The source documents should be retained if redistribution is permitted. Otherwise, provide instructions for obtaining them from their authoritative sources.

---

## 7. Knowledge-Base Preparation

TALA uses a document-ingestion and embedding pipeline to prepare NASA literature for retrieval.

The knowledge-base construction script is:

```bash
python scripts/build_knowledge_base.py
```

This script performs the configured knowledge-base preparation process, including document processing, chunking, embedding, and vector-store construction.

### Embedding Configuration

| Setting              | Value                  |
| -------------------- | ---------------------- |
| Embedding model      | `qwen3-embedding:0.6b` |
| Embedding provider   | Ollama                 |
| Embedding dimensions | 1024                   |
| Vector database      | ChromaDB               |
| Collection           | `tala_nasa_knowledge`  |
| Similarity metric    | Cosine distance        |

The ingestion process uses page-aware document extraction, paragraph-based chunking, and source metadata to preserve document identity and page references.

The recorded ingestion result was:

| Metric              | Result |
| ------------------- | -----: |
| Source PDFs         |      6 |
| Pages processed     |    839 |
| Chunks created      |  1,341 |
| Empty pages         |      0 |
| Processing failures |      0 |
| Processing warnings |      0 |

The resulting vector database is stored at:

```text
data/knowledge_base/chroma_db/
```

The ChromaDB directory is generated data. It can be rebuilt from the source PDFs using the knowledge-base construction script and the configured embedding model.

---

## 8. Running the Complete Pipeline

For full reproduction, execute the stages in their dependency order.

### Step 1 — Build the NASA Knowledge Base

```bash
python scripts/build_knowledge_base.py
```

This prepares the NASA document embeddings and ChromaDB collection.

### Step 2 — Run Pose Estimation

```bash
python scripts/run_pose_estimation.py
```

This processes the exercise video and extracts pose landmarks.

Expected outputs include:

```text
results/pose_estimation/
├── extraction_report.json
├── pose_landmarks.csv
├── video_metadata.json
└── visualizations/
    └── pose_overlay.mp4
```

### Step 3 — Run OpenSim Kinematic Reconstruction

```bash
python scripts/run_kinematic_reconstruction.py
```

This maps the pose observations to the musculoskeletal model and performs inverse kinematics.

Expected outputs include:

```text
results/kinematic_reconstruction/
├── mapped_coordinates.csv
├── ik_report.json
├── marker_fitting_errors.csv
├── opensim_motion.mot
├── pose_targets.trc
└── scaled_model.osim
```

### Step 4 — Run Gravity Analysis

```bash
python scripts/run_gravity_analysis.py
```

This calculates COM-related gravitational potential energy for the configured gravity environments.

### Step 5 — Run Movement Analysis

```bash
python scripts/run_movement_analysis.py
```

This analyzes the reconstructed movement, including movement phases and repetition characteristics.

### Step 6 — Run the Unified TALA Pipeline

```bash
python scripts/run_tala.py --question "Explain the movement findings and their relevance to astronaut exercise."
```

The unified pipeline combines the available analysis artifacts and NASA literature retrieval into a consolidated output.

**Important:** Each upstream stage must produce its expected input files before the next dependent stage is run.

---

## 9. Running the Unified Pipeline Independently

For a quick demonstration, the upstream pose-estimation, reconstruction, gravity, and movement stages do not need to be rerun if their required outputs are already available.

The unified pipeline consumes these files:

```text
results/
├── kinematic_reconstruction/
│   ├── mapped_coordinates.csv
│   └── ik_report.json
│
├── gravity_analysis/
│   └── gravity_analysis.json
│
└── movement_analysis/
    └── movement_analysis.json
```

It also requires the configured NASA knowledge base to be available for RAG functionality.

Run:

```bash
python scripts/run_tala.py --question "Explain the movement findings and their relevance to astronaut exercise."
```

The unified result is written to:

```text
results/tala_output.json
```

The unified pipeline does not automatically regenerate pose estimation, OpenSim reconstruction, gravity analysis, or movement analysis. It consumes their existing outputs.

---

## 10. Assessment and Baseline Behavior

The assessment module combines:

- Daily check-in information, when provided.
- Bilateral hip, knee, and ankle observations.
- Kinematic reconstruction-quality information.
- Gravity-analysis context.
- Movement-related observations.
- Contextual insights and limitations.

The assessment accepts partial daily check-ins. Missing values are retained as missing rather than being inferred to be normal.

### Baseline Availability

TALA includes a session-baseline component for comparing compatible historical movement sessions.

A baseline is only considered when a compatible historical session is available for the same exercise and analysis scope.

Consequently:

- A first session may report that a baseline is unavailable.
- A single session does not establish a historical baseline.
- Reprocessing the same recording does not create a new independent session.
- Missing historical information is not replaced with fabricated reference values.

Baseline availability is therefore dependent on the available session history, not a failure of the assessment pipeline.

---

## 11. Output Structure

The primary output directories are:

| Directory                           | Contents                                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| `results/pose_estimation/`          | Extracted landmarks, video metadata, extraction report, and pose visualization. |
| `results/kinematic_reconstruction/` | OpenSim motion, mapped coordinates, marker-fitting diagnostics, and IK report.  |
| `results/gravity_analysis/`         | Gravity-analysis report and energy time series.                                 |
| `results/movement_analysis/`        | Movement-analysis report and phase segmentation.                                |
| `results/assessment/`               | Assessment output.                                                              |
| `results/baselines/`                | Local historical session information.                                           |
| `results/`                          | Unified TALA output.                                                            |

The primary integrated output is:

```text
results/tala_output.json
```

This file contains the consolidated results generated by the unified pipeline.

---

## 12. Testing

Run the automated test suite from the project root:

```bash
pytest -q
```

The latest recorded project test run completed with:

```text
8 passed in 20.76s
```

This result reflects the test suite and code state at the time of that run. Re-run the tests after installing dependencies or modifying the project.

### Diagnostic Scripts

The repository also includes scripts for checking individual components:

| Script                      | Purpose                                              |
| --------------------------- | ---------------------------------------------------- |
| `test_gemini_connection.py` | Checks Gemini API configuration and connectivity.    |
| `test_rag_retrieval.py`     | Checks retrieval from the local NASA knowledge base. |
| `test_rag_generation.py`    | Checks evidence-grounded answer generation.          |
| `examine_movement.py`       | Inspects movement-analysis results.                  |
| `run_assessment_example.py` | Runs the assessment module independently.            |

These are diagnostic or standalone scripts, not replacements for the complete pipeline.

---

## 13. Reproducibility and Version-Control Policy

The repository distinguishes between source code, required source assets, generated databases, and local outputs.

### Recommended Git Inclusion

| File or Directory                         | Recommended Treatment                                         |
| ----------------------------------------- | ------------------------------------------------------------- |
| `src/tala/`                               | Include                                                       |
| `scripts/`                                | Include                                                       |
| `configs/`                                | Include                                                       |
| `tests/`                                  | Include                                                       |
| `docs/`                                   | Include                                                       |
| `pyproject.toml`                          | Include                                                       |
| `requirements.txt`                        | Include if maintained consistently                            |
| `README.md`                               | Include                                                       |
| `.env.example`                            | Include                                                       |
| `.env`                                    | Exclude                                                       |
| `.venv/`                                  | Exclude                                                       |
| `data/knowledge_base/raw/`                | Include if redistribution is permitted                        |
| `data/knowledge_base/chroma_db/`          | Rebuild locally                                               |
| `models/`                                 | Include or provide documented downloads, subject to licensing |
| `data/raw/squat_segment.mp4`              | Include only if redistribution is permitted                   |
| `results/baselines/session_history.jsonl` | Exclude                                                       |
| `results/tala_output.json`                | Exclude as generated output                                   |
| Other generated `results/` files          | Exclude, except selected demo fixtures if required            |

### Quick-Demo Fixtures

If the repository is intended to support a quick demonstration immediately after cloning, preserve these four input artifacts:

```text
results/kinematic_reconstruction/mapped_coordinates.csv
results/kinematic_reconstruction/ik_report.json
results/gravity_analysis/gravity_analysis.json
results/movement_analysis/movement_analysis.json
```

These allow the unified pipeline to run without first repeating the upstream analyses.

The ChromaDB vector database should remain rebuildable from the source documents rather than being treated as source code.

### Secrets

Never commit:

- `.env`
- Gemini API keys
- Private credentials
- Local environment files

If a real API key has previously been committed, revoke or rotate it. Adding `.env` to `.gitignore` does not remove an already tracked secret from Git history.

---

## 14. Known Limitations

TALA is an exploratory prototype, and its outputs must be interpreted within the limitations of its data and methods.

### Computer Vision

- Pose estimation is based on monocular video.
- The extracted landmarks are visual estimates, not calibrated motion-capture measurements.
- Camera perspective and occlusion can affect landmark accuracy.
- Depth information cannot be directly established from a single uncalibrated camera view.
- The available footage does not provide ground-truth 3D joint positions.

### OpenSim Kinematic Reconstruction

- The 3D reconstruction is model-constrained and informed by monocular visual observations.
- It should not be interpreted as direct camera-measured 3D motion.
- Model scaling, marker mapping, and inverse-kinematics assumptions affect the reconstructed coordinates.
- Unobserved degrees of freedom may be constrained or held at selected values.
- No independent motion-capture ground truth is available for validation.

### Biomechanical and Gravity Analysis

- Gravity comparisons use the same fixed reconstructed trajectory under different gravity constants.
- They do not simulate physiological adaptation or behavioral changes in lunar or Martian environments.
- Gravitational potential energy is a simplified physical quantity, not a direct measurement of muscle effort or exercise effectiveness.
- The current pipeline does not calculate muscle forces, joint moments, muscle activation, tissue stress, or injury risk.

### Assessment and Baselines

- Assessment outputs are exploratory observations, not clinical evaluations.
- Missing check-in values are not inferred.
- Baseline comparisons depend on compatible historical sessions.
- The system does not establish clinical thresholds or diagnose medical conditions.

### Retrieval-Augmented Generation

- Generated explanations depend on the retrieved NASA passages and the language model.
- Retrieval similarity is not a guarantee of factual correctness.
- The knowledge base is limited to its included source documents.
- RAG provides literature-grounded context; it does not validate or modify numerical analysis results.
- Generated text should be interpreted alongside the cited source material.

---

## 15. Reproducibility Checklist

Before considering a fresh installation ready for evaluation:

- [ ] Python 3.11 environment created.
- [ ] Project dependencies installed.
- [ ] Required OpenSim model available.
- [ ] MediaPipe Pose Landmarker model available.
- [ ] Source exercise video available.
- [ ] Ollama installed and running.
- [ ] `qwen3-embedding:0.6b` available.
- [ ] Gemini API key configured in `.env`.
- [ ] NASA source PDFs available.
- [ ] Knowledge base successfully built.
- [ ] Required upstream artifacts available or regenerated.
- [ ] Unified TALA pipeline executes successfully.
- [ ] `results/tala_output.json` generated.
- [ ] Automated tests pass.

For a stronger reproducibility check, test the quick-demonstration workflow from a clean clone rather than relying only on the original development environment.

---

## 16. Research Scope Statement

TALA demonstrates an integrated exploratory workflow connecting markerless pose estimation, model-based OpenSim kinematic reconstruction, movement and gravity-related analysis, and NASA-literature-grounded contextual retrieval.

The current implementation focuses on the feasibility of connecting these components into a reproducible backend pipeline. It does not establish clinical validity, quantify astronaut physiological adaptation, or replace calibrated biomechanical measurement systems.

The prototype is intended to demonstrate how exercise-motion observations and relevant spaceflight literature can be brought together in a structured analytical workflow, while preserving the distinction between estimated measurements, model-derived quantities, and literature-based contextual explanations.

---

## 17. Acknowledgment and Source Documentation

The knowledge base uses NASA human-health, human-system, and spaceflight-related documentation. The source manifest and RAG methodology documentation provide additional information about document ingestion and retrieval.

For the research context and scientific interpretation, consult the accompanying TALA research paper.
