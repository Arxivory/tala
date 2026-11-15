# TALA Backend
## An Exploratory Framework for Astronaut Exercise Movement Analysis

TALA (Twin-based Astronaut Life Analytics) is an exploratory research framework integrating computer vision, OpenSim-based kinematic reconstruction, biomechanical and movement analysis, and NASA-literature-grounded retrieval-augmented generation (RAG).

The backend processes exercise video to extract human pose landmarks, uses these observations to inform musculoskeletal model-based kinematic reconstruction, derives movement and gravity-related quantities, and generates contextual explanations grounded in NASA human-health and spaceflight literature.

TALA is a research and demonstration prototype. Its outputs are exploratory and are not intended for clinical diagnosis, medical decision-making, or evaluation of astronaut fitness.

---

## 1. Project Overview

The TALA backend consists of four principal processing stages:

1. **Computer Vision:** Extracts human pose landmarks from exercise video using MediaPipe Pose Landmarker.
2. **OpenSim Kinematic Reconstruction:** Maps visual observations to a musculoskeletal model and performs inverse kinematics to estimate model-constrained joint kinematics.
3. **Biomechanical and Movement Analysis:** Analyzes reconstructed movement, identifies movement phases and repetitions, calculates whole-body center of mass (COM), and evaluates relative gravitational potential energy under different gravity constants.
4. **NASA Literature-Grounded RAG:** Retrieves relevant passages from NASA documents and generates contextual explanations using a language model.

These components are integrated through a unified assessment pipeline combining movement observations, reconstruction-quality information, gravity context, available daily check-in information, and retrieved NASA literature.

### Research Objectives

The backend investigates the feasibility of:

- Extracting usable pose landmarks from astronaut exercise footage.
- Using monocular visual observations to inform OpenSim-based kinematic reconstruction.
- Deriving movement-related quantities from reconstructed kinematics.
- Comparing gravitational potential energy for a fixed reconstructed movement trajectory under different gravity constants.
- Retrieving NASA literature and generating evidence-grounded contextual explanations.
- Integrating these components into a reproducible analytical workflow.

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

| Component | Responsibility |
|---|---|
| Pose Estimation | Extracts image-based body landmarks from exercise video. |
| Kinematic Reconstruction | Uses visual targets and an OpenSim musculoskeletal model to estimate joint kinematics through inverse kinematics. |
| Movement Analysis | Identifies movement phases and repetitions and summarizes movement characteristics. |
| Gravity Analysis | Calculates COM-related gravitational potential energy for specified gravity constants. |
| Assessment | Combines movement observations, reconstruction quality, gravity context, and available daily check-in information. |
| NASA RAG | Retrieves relevant NASA literature and generates contextual explanations. |

The RAG component provides supporting literature context. It does not modify numerical calculations or independently determine assessment flags.

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

Generated outputs and local vector databases are produced during execution and may not be included in the source repository.

---

## 4. Technical Requirements

### Software and Dependencies

- Python 3.11
- Git
- Ollama
- Google Gemini API key
- OpenSim-compatible Python environment
- Dependencies declared in `pyproject.toml` and `requirements.txt`

The development environment used Python 3.11.9.

### Environment Installation

The following commands illustrate the Windows PowerShell setup:

```powershell
git clone <REPOSITORY_URL>
cd tala-backend

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
```

The required Python dependencies are defined in the project configuration.

---

## 5. Environment Configuration

### Google Gemini

Google Gemini is used for natural-language generation based on retrieved NASA literature.

Create a `.env` file in the repository root:

```env
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

The `.env.example` file provides a configuration template. API keys and other credentials must remain private and must not be committed to version control.

### Ollama Embeddings

Ollama provides local text embeddings for the NASA knowledge base.

The configured embedding model is:

```text
qwen3-embedding:0.6b
```

Model installation:

```bash
ollama pull qwen3-embedding:0.6b
```

The Ollama service must be available during knowledge-base construction and retrieval.

### Model and Service Responsibilities

| Model / Service | Responsibility |
|---|---|
| Qwen3 Embedding 0.6B through Ollama | Converts NASA document chunks and queries into vector embeddings. |
| ChromaDB | Stores and retrieves embedded document chunks. |
| Google Gemini | Generates contextual answers using retrieved evidence. |

Embedding and language generation are separate operations. Ollama handles embedding generation, while Gemini handles natural-language generation.

---

## 6. Models and Input Data

### OpenSim Musculoskeletal Model

**File:** `models/opensim/RajagopalLaiUhlrich2023.osim`

The Rajagopal musculoskeletal model is used for model-based kinematic reconstruction.

### MediaPipe Pose Landmarker

**File:** `models/pose_landmarker/pose_landmarker_full.task`

The MediaPipe model extracts body landmarks from the input exercise video.

### Exercise Video

**File:** `data/raw/squat_segment.mp4`

The exercise video serves as the input to the computer-vision pipeline.

### NASA Knowledge-Base Documents

The RAG knowledge base contains six NASA-related documents:

1. NASA-STD-3001 Volume 1 — Crew Health.
2. NASA-STD-3001 Volume 2.
3. OCHMO-HB-004 — Anthropometry, Biomechanics, and Strength.
4. OCHMO-TB-031 — Exercise Overview: Astronaut Physiological Deconditioning and Exercise Prescription Countermeasures in Spaceflight.
5. HRP Evidence Report — Reduced Cardiorespiratory and Musculoskeletal Fitness.
6. HRP Evidence Report — Sleep Loss, Circadian Desynchronization, and Work Overload.

The source PDFs are organized under:

```text
data/knowledge_base/raw/
```

Their source information and associated metadata are documented in `source_manifest.json`.

---

## 7. NASA Knowledge-Base Construction

TALA uses a document-ingestion and embedding pipeline to prepare NASA literature for semantic retrieval.

The knowledge-base construction script is:

```bash
python scripts/build_knowledge_base.py
```

The script processes the configured source documents, extracts their text, divides the content into retrievable chunks, generates embeddings, and constructs the ChromaDB vector collection.

### Embedding and Retrieval Configuration

| Setting | Value |
|---|---|
| Embedding model | `qwen3-embedding:0.6b` |
| Embedding provider | Ollama |
| Embedding dimensions | 1024 |
| Vector database | ChromaDB |
| Collection | `tala_nasa_knowledge` |
| Similarity metric | Cosine distance |

The ingestion workflow uses page-aware extraction, paragraph-based chunking, and source metadata to preserve document identity and page references.

### Recorded Knowledge-Base Ingestion Results

| Metric | Result |
|---|---:|
| Source PDFs | 6 |
| Pages processed | 839 |
| Chunks created | 1,341 |
| Empty pages | 0 |
| Processing failures | 0 |
| Processing warnings | 0 |

The generated vector database is stored at:

```text
data/knowledge_base/chroma_db/
```

The database is a generated artifact and can be reconstructed from the source PDFs using the knowledge-base construction script and configured embedding model.

---

## 8. Pipeline Execution

The backend supports execution of individual processing stages as well as a unified pipeline.

### 8.1 Build the NASA Knowledge Base

```bash
python scripts/build_knowledge_base.py
```

This generates the document embeddings and ChromaDB collection required for NASA literature retrieval.

### 8.2 Run Pose Estimation

```bash
python scripts/run_pose_estimation.py
```

This processes the exercise video and extracts pose landmarks.

Expected outputs:

```text
results/pose_estimation/
├── extraction_report.json
├── pose_landmarks.csv
├── video_metadata.json
└── visualizations/
    └── pose_overlay.mp4
```

### 8.3 Run OpenSim Kinematic Reconstruction

```bash
python scripts/run_kinematic_reconstruction.py
```

This maps the extracted pose observations to the musculoskeletal model and performs inverse kinematics.

Expected outputs:

```text
results/kinematic_reconstruction/
├── mapped_coordinates.csv
├── ik_report.json
├── marker_fitting_errors.csv
├── opensim_motion.mot
├── pose_targets.trc
└── scaled_model.osim
```

### 8.4 Run Gravity Analysis

```bash
python scripts/run_gravity_analysis.py
```

This calculates COM-related gravitational potential energy for the configured gravity environments.

### 8.5 Run Movement Analysis

```bash
python scripts/run_movement_analysis.py
```

This analyzes reconstructed movement, including movement phases and repetition characteristics.

### 8.6 Run the Unified TALA Pipeline

```bash
python scripts/run_tala.py --question "Explain the movement findings and their relevance to astronaut exercise."
```

The unified pipeline combines the available analysis artifacts and NASA literature retrieval into a consolidated output.

The upstream processing stages must produce their required input files before dependent stages can execute.

---

## 9. Unified Pipeline and Existing Analysis Artifacts

The unified TALA pipeline can operate using previously generated analysis artifacts. It does not automatically repeat all upstream processing stages.

The required analytical inputs include:

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

The configured NASA knowledge base must also be available for RAG functionality.

Execution:

```bash
python scripts/run_tala.py --question "Explain the movement findings and their relevance to astronaut exercise."
```

The consolidated output is written to:

```text
results/tala_output.json
```

This workflow supports demonstration and assessment using previously generated analysis results without rerunning the computationally intensive upstream stages.

---

## 10. Assessment and Baseline Behavior

The assessment module combines the following information:

- Daily check-in information, when provided.
- Bilateral hip, knee, and ankle observations.
- Kinematic reconstruction-quality information.
- Gravity-analysis context.
- Movement-related observations.
- Contextual insights and limitations.

Partial daily check-ins are supported. Missing values remain explicitly missing rather than being inferred to represent normal conditions.

### Historical Session Baselines

TALA includes a session-baseline component for comparing compatible historical movement sessions.

A baseline is considered available only when a compatible historical session exists for the same exercise and analysis scope.

The implementation follows these principles:

- A first session may report that a baseline is unavailable.
- A single session does not establish a historical baseline.
- Reprocessing the same recording does not create a new independent session.
- Missing historical information is not replaced with fabricated reference values.

Baseline availability therefore depends on the available compatible session history.

---

## 11. Output Structure

| Directory | Contents |
|---|---|
| `results/pose_estimation/` | Extracted landmarks, video metadata, extraction report, and pose visualization. |
| `results/kinematic_reconstruction/` | OpenSim motion, mapped coordinates, marker-fitting diagnostics, and IK report. |
| `results/gravity_analysis/` | Gravity-analysis report and energy time series. |
| `results/movement_analysis/` | Movement-analysis report and phase segmentation. |
| `results/assessment/` | Assessment output. |
| `results/baselines/` | Local historical session information. |
| `results/` | Unified TALA output. |

The primary integrated output is:

```text
results/tala_output.json
```

It contains the consolidated results produced by the unified assessment pipeline.

---

## 12. Testing and Validation

The automated test suite can be executed from the project root:

```bash
pytest -q
```

The latest recorded test run completed with:

```text
8 passed in 20.76s
```

This result represents the test suite and implementation state at the time of execution.

### Diagnostic Scripts

| Script | Purpose |
|---|---|
| `test_gemini_connection.py` | Checks Gemini API configuration and connectivity. |
| `test_rag_retrieval.py` | Checks retrieval from the local NASA knowledge base. |
| `test_rag_generation.py` | Checks evidence-grounded answer generation. |
| `examine_movement.py` | Inspects movement-analysis results. |
| `run_assessment_example.py` | Executes the assessment module independently. |

These scripts support component-level inspection and testing.

---

## 13. Reproducibility and Version Control

The repository separates source code, required source assets, generated databases, and local outputs.

### Recommended Version-Control Policy

| File or Directory | Treatment |
|---|---|
| `src/tala/` | Version-controlled source code. |
| `scripts/` | Version-controlled scripts. |
| `configs/` | Version-controlled configuration files. |
| `tests/` | Version-controlled automated tests. |
| `docs/` | Version-controlled documentation. |
| `pyproject.toml` | Version-controlled project configuration. |
| `requirements.txt` | Version-controlled dependency list, if maintained consistently. |
| `README.md` | Version-controlled documentation. |
| `.env.example` | Version-controlled configuration template. |
| `.env` | Excluded; contains private credentials. |
| `.venv/` | Excluded; local Python environment. |
| `data/knowledge_base/raw/` | Include subject to source-document redistribution permissions. |
| `data/knowledge_base/chroma_db/` | Generated locally from source documents. |
| `models/` | Include or provide documented downloads, subject to licensing. |
| `data/raw/squat_segment.mp4` | Include only if redistribution is permitted. |
| `results/baselines/session_history.jsonl` | Excluded; local session history. |
| `results/tala_output.json` | Generated output. |
| Other generated `results/` files | Excluded unless selected as demonstration fixtures. |

### Demonstration Artifacts

The following files can be retained as demonstration fixtures when a quick unified-pipeline execution is desired:

```text
results/kinematic_reconstruction/mapped_coordinates.csv
results/kinematic_reconstruction/ik_report.json
results/gravity_analysis/gravity_analysis.json
results/movement_analysis/movement_analysis.json
```

These allow the unified pipeline to consume previously generated analysis outputs without repeating upstream processing.

The ChromaDB database remains reproducible from the source documents and embedding configuration.

### Credential Security

API keys, private credentials, and local environment files must not be committed to the repository.

If a credential is exposed in version-control history, it should be revoked or rotated. Adding a file to `.gitignore` does not remove previously committed secrets from Git history.

---

## 14. Scientific and Technical Limitations

TALA is an exploratory research prototype. Its outputs must be interpreted within the limitations of its data, models, and analytical methods.

### Computer Vision

- Pose estimation is based on monocular video.
- Extracted landmarks are visual estimates, not calibrated motion-capture measurements.
- Camera perspective and occlusion can affect landmark accuracy.
- Depth information cannot be directly established from a single uncalibrated camera view.
- The available footage does not provide ground-truth 3D joint positions.

### OpenSim Kinematic Reconstruction

- The 3D reconstruction is model-constrained and informed by monocular visual observations.
- It should not be interpreted as direct camera-measured 3D motion.
- Model scaling, marker mapping, and inverse-kinematics assumptions affect reconstructed coordinates.
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

- Generated explanations depend on retrieved NASA passages and the language model.
- Retrieval similarity does not guarantee factual correctness.
- The knowledge base is limited to its included source documents.
- RAG provides literature-grounded context; it does not validate or modify numerical analysis results.
- Generated explanations should be interpreted alongside their cited source material.

---

## 15. Reproducibility Checklist

A complete local execution requires:

- Python 3.11 environment.
- Installed project dependencies.
- OpenSim musculoskeletal model.
- MediaPipe Pose Landmarker model.
- Source exercise video.
- Ollama installation and running service.
- `qwen3-embedding:0.6b` embedding model.
- Gemini API key configured in `.env`.
- NASA source PDFs.
- Successfully constructed knowledge base.
- Required upstream analysis artifacts, either generated or provided as demonstration fixtures.
- Successful unified pipeline execution.
- Generated `results/tala_output.json`.
- Passing automated tests.

---

## 16. Research Scope

TALA demonstrates an integrated exploratory workflow connecting markerless pose estimation, model-based OpenSim kinematic reconstruction, movement and gravity-related analysis, and NASA-literature-grounded contextual retrieval.

The implementation investigates the feasibility of connecting these components into a reproducible backend research pipeline. It does not establish clinical validity, quantify astronaut physiological adaptation, or replace calibrated biomechanical measurement systems.

The framework demonstrates how exercise-motion observations and relevant spaceflight literature can be brought together in a structured analytical workflow while preserving the distinction between visual estimates, model-derived quantities, and literature-based contextual explanations.

---

## 17. Documentation and References

Additional implementation and methodological documentation is available in the repository:

- `docs/architecture.md` — System architecture and component relationships.
- `docs/methodology.md` — Processing methodology and analytical procedures.
- `docs/limitations.md` — Technical and scientific limitations.
- `docs/rag_methodology.md` — NASA document ingestion, embedding, retrieval, and generation methodology.
- `data/knowledge_base/source_manifest.json` — Knowledge-base source documentation.

The accompanying TALA research paper provides the broader research context, methodology, experimental observations, and interpretation of results.