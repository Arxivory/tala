# TALA Research

Starter scaffold for the active TALA research pipeline:

1. **Pose estimation** — MediaPipe pose landmarks from recorded exercise video (Phase 3.3).
2. **Kinematic reconstruction** — landmark-to-OpenSim full-body inverse kinematics (Phase 3.10).
3. **Scientific context (planned)** — a small RAG component for evidence-grounded explanations.

This ZIP is a base structure, not a completed refactor. The original Phase 3.3 and Phase 3.10 scripts have not yet been migrated into the placeholder modules.

## Scope

The active core covers video-based pose estimation, landmark preprocessing, OpenSim model scaling and marker mapping, inverse kinematics, coordinate export and reconstruction diagnostics.

Neural surrogate inference, inverse dynamics, joint-load estimation from video, and claims of independent motion-capture validation are outside this active scope.

RAG is an explanatory evidence-retrieval component. It must not modify biomechanical calculations, invent measurements, diagnose users, or independently decide exercise readiness.

## Required local assets

These are not included in the ZIP. Copy them locally:

| Asset | Expected path |
|---|---|
| Exercise video | `data/raw/astronaut_routine/Squat2.mp4` |
| MediaPipe task model | `models/pose_landmarker/pose_landmarker_full.task` |
| OpenSim model | `models/opensim/RajagopalLaiUhlrich2023.osim` |

Preserve any OpenSim geometry directory structure required by the model.

## Setup

Create and activate a Python 3.11+ virtual environment, then:

```bash
python -m pip install --upgrade pip
pip install -e .
```

OpenSim Python bindings are platform/environment-specific and are intentionally not pinned in the package dependencies. Configure OpenSim separately for your environment.

Optional RAG dependencies:

```bash
pip install -e ".[rag]"
```

## Current status

- [x] Initial package and directory scaffold
- [x] Phase 3.3 YAML copied from the current implementation (output path adapted to the clean repository)
- [x] Shared JSON/CSV serialization helpers
- [ ] Refactor Phase 3.3
- [ ] Refactor Phase 3.10
- [ ] Compare refactored outputs with original scripts on the same video
- [ ] Implement RAG ingestion, retrieval, generation and citations
- [ ] Define frontend/demo export contract

Keep the original research repository unchanged. During refactoring, preserve algorithms, assumptions, report fields and output filenames unless a deliberate change is documented and tested.
