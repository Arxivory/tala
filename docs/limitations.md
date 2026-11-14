# Limitations

- MediaPipe world landmarks are estimates, not camera-calibrated motion capture.
- A single camera introduces depth ambiguity.
- Pose and body-segment scale factors are estimated from monocular landmarks.
- Pelvis camera-depth translation is held at zero in the current method.
- Interpolated samples are estimates, not direct observations.
- Neutralized unobserved limbs are assumptions, not recovered movement.
- Virtual wrist markers are attached to hand origins; wrist flexion/deviation
  are not reconstructed by those markers.
- Marker-fitting residuals compare the fitted model with the transformed pose
  targets used in IK; they are not independent anatomical validation.
- The pipeline is kinematic; it does not compute inverse dynamics, muscle
  forces or joint loading.
