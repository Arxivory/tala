# Architecture

```text
Recorded video
    |
    v
Pose estimation (Phase 3.3)
    | pose_landmarks.csv + video_metadata.json + extraction_report.json
    v
Kinematic reconstruction (Phase 3.10)
    | scaled_model.osim + pose_targets.trc + opensim_motion.mot
    | mapped_coordinates.csv + marker_fitting_errors.csv + ik_report.json
    v
Structured movement context
    |--------------------------|
    v                          v
Rules/context layer           RAG service
    |                          |
    |--------------------------|
               |
               v
      Evidence-linked insights
```

`pose_estimation` owns video decoding, inference and extraction reporting.
`kinematic_reconstruction` owns landmark preprocessing, scaling, transforms,
marker mapping, OpenSim IK and diagnostics. `rag` owns source ingestion,
retrieval and evidence-grounded explanation. `common` contains small shared
utilities only.

RAG consumes structured outputs; it does not call or modify OpenSim or
MediaPipe calculations.
