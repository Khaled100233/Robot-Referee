# Configuration Files Directory

Store YAML and JSON configuration files here for different deployment scenarios.

## Examples

### deployment_config.yaml
```yaml
environment: production
model:
  pose: models/custom_pose.pt
  detect: models/custom_detect.pt
inference:
  confidence: 0.6
  device: cuda
  batch_size: 1
```

### thresholds.yaml
```yaml
cases:
  unnatural_arm:
    angle_threshold: 135
    min_confidence: 0.6
  reaction_time:
    max_time_ms: 500
  deflection:
    min_trajectory_change: 15
```

Use these configs to manage different deployment environments without changing code.
