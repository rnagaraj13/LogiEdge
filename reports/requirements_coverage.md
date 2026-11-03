# LogiEdge requirement coverage matrix

This matrix links each assignment requirement to the corresponding repository artifact(s). It is intended to help you verify that no section was skipped when preparing final submission evidence.

| Assignment section | Evidence in repository |
|---|---|
| A1 Constraint analysis (350–500 words) | `scenario_architecture/constraint_analysis.md` |
| A2 System architecture diagram | `scenario_architecture/system_architecture.svg` (+ editable `.mmd`) |
| B1 Constraint Triangle and hardware choice | `hardware/hardware_justification.md` |
| B2 Arithmetic Intensity + Roofline classification | `hardware/hardware_justification.md` |
| C1 Simulator with `--anomaly {none,temp_drift,vibration,combined}` | `data_pipeline/simulator.py` |
| C2 Filtering, windows, six features, frozen stats, 3σ experiment | `inference/preprocessing.py`, `data_pipeline/generate_dataset.py`, `training/train_model.py` |
| C3 Feature-level fusion justification | `data_pipeline/mqtt_architecture.md` |
| D1 Dataset generation and model training gate (88%) | `data_pipeline/generate_dataset.py`, `training/train_model.py` |
| D2 Docker service + MODEL_PATH + topic publish + layer-cache procedure | `inference/Dockerfile`, `inference/inference_service.py`, `inference/README.md` |
| D3 Ten-stage pipeline mapping | `reports/pipeline_mapping.md` |
| E1 PSI with 300 reference windows, 100 rolling, 60s checks, thresholds | `monitoring/build_reference.py`, `monitoring/drift_monitor.py`, `monitoring/psi.py` |
| E2 Ansible playbook with exactly seven tasks | `deployment/logibridge_deploy.yml` |
| E3 OTA strategy and cost comparison | `deployment/ota_strategy.md`, `deployment/ota_cost.py` |
| F1 Three variants M1/M2/M3 | `training/train_model.py`, `training/convert_ptq.py`, `training/prune_quantise.py` |
| F2 Five-metric benchmark + Pareto chart | `optimisation/benchmark.py` |
| F3 Deployment recommendation with SLA and Class 2 recall gate | `optimisation/deployment_recommendation.md` |
| Deliverable: demo link file | `demo/demo_video_link.txt` |
| Deliverable: report scaffold (2,500–3,000 words) | `reports/final_report_template.md`, `reports/check_report.py` |

## Final submission checks you still must perform

1. Run all scripts end-to-end in your target environment and capture **measured** outputs.
2. Export the final report to PDF as `GROUPNO_LogiEdge_Final.pdf`.
3. Record your own demo and verify link accessibility in incognito mode.
4. Add the instructor as collaborator to the private repository before submission.
