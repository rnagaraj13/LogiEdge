# E2 — Ansible rollout and idempotency

Install the `community.docker` collection on the deployment controller:

```bash
ansible-galaxy collection install community.docker
```

Set `model_source`, `reference_source`, `truck_id`, and registry values with `--extra-vars` or an uncommitted vault/inventory. The target must have Docker and a local registry reachable. Stage `training_stats.npy` at `/opt/logibridge/training_stats.npy` as part of the node's initial provisioning; the seven-task playbook intentionally implements the seven tasks listed in the assignment.

Run twice against the same node and immutable inputs:

```bash
ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml
ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml
```

Capture both actual PLAY RECAPs in the demo. On the second run, verify the recap says `changed=0`. The stop task is gated on changed model/reference files, the Docker image pull and container state are declarative, and the final wait/health inspection is marked unchanged. If a registry tag is mutable and points to a new digest, an image update is an actual change and `changed=0` is not expected; use an immutable image digest when demonstrating idempotency.

There are exactly seven entries under `tasks:` in `logibridge_deploy.yml`. The fourth task only stops a running container when one of the copied artifacts changed, preventing an unnecessary stop/start on the second identical run.
