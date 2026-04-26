# Anti-Leakage Policy

This document defines the threat model and initial control policy for preventing miner
overfitting, memorization, or direct leakage of validator-private human data in Poker44.

## Threat Model

Poker44 validators evaluate miners using private human hands plus generated bot hands.
The main failure modes are:

- a miner trains on leaked validator-private human data;
- a miner memorizes specific hands or chunks instead of learning transferable behavior;
- a miner hardcodes lookup logic against known evaluation payloads;
- a miner overfits to public benchmark artifacts and fails to generalize to live validator data.

Open source alone does not eliminate these risks. It improves auditability, but the subnet
still needs private evaluation discipline and explicit anti-leakage controls.

## Policy Goals

- keep validator-private human data private;
- maximize miner generalization to unseen hands and windows;
- make cheating more detectable and more expensive;
- create a clear compliance standard for miners without breaking the current evaluation loop.

## Immediate Controls

These controls fit the current remote-inference architecture.

### 1. Open-Source Manifest With Data Disclosure

Miners should publish a `model_manifest` that includes:

- repo URL
- repo commit or tag
- model name and version
- framework and license
- training data statement
- training data sources
- private data attestation
- artifact hash when a checkpoint exists

This is not proof, but it forces a public claim about how the miner was built.

### 2. Public/Private Dataset Separation

Poker44 should continue to enforce a hard boundary:

- public benchmark for miner training and reference only;
- validator-private human data for live evaluation only.

The public benchmark must never be treated as a proxy for production validator data.

### 3. Dynamic Private Evaluation Windows

Validators should keep evaluating on rotating windows of private human data and generated bot
data. A miner that memorizes one static corpus should not be rewarded for long.

### 4. Generalization-First Reward Interpretation

Weight-setting should remain based on performance over windows, not on a single lucky cycle.
A miner that spikes on one batch and degrades on fresh windows should lose weight over time.

### 5. Manifest Compliance Tier

Operationally, Poker44 should distinguish:

- `transparent`: open-source manifest present with data disclosure fields;
- `opaque`: no manifest or incomplete disclosure.

This tier should initially affect visibility and trust, not scoring.

Current implementation:

- validators persist model manifests by UID;
- validators persist a separate compliance registry with `transparent` / `opaque` status;
- validators log suspicion events when manifests are missing or incomplete;
- validators track served chunk fingerprints to monitor repeated exposure over time.

## Recommended Near-Term Controls

These are the next controls worth implementing.

### 6. Canary Hands

Inject a small number of validator-controlled canary hands or canary chunks that are never
published in the public benchmark.

Use canaries to detect:

- suspiciously confident predictions on unique patterns;
- abrupt performance asymmetries that suggest memorization;
- repeated exact behavior against synthetic sentinel examples.

Canaries should be rotated and versioned privately.

### 7. Duplicate and Near-Duplicate Screening

Validators should fingerprint sanitized hands and chunks to reduce overlap between:

- live evaluation windows;
- previously served batches;
- public benchmark artifacts.

The aim is not absolute uniqueness, but minimizing repeated evaluation payloads that miners
could memorize over time.

### 8. Holdout Regimes

Maintain distinct evaluation pools:

- rolling live pool;
- longer-term hidden holdout pool;
- optional challenge pool reserved for audits.

If a miner performs unusually well on one pool and weakly on another, that is a leakage signal.

### 9. Suspicion Logging

Track structured suspicion indicators per UID:

- manifest missing or inconsistent;
- implausibly sharp step-change in score;
- strong canary sensitivity;
- overlap anomalies;
- unstable generalization between windows.

Suspicion should not immediately zero a miner without review, but it should create an audit trail.

## Stronger Future Controls

These controls require larger architectural changes.

### 10. Artifact Submission and Local Validator Evaluation

Move from pure remote inference toward artifact-based verification:

- miner declares model hash;
- validator downloads artifact;
- validator runs the declared artifact locally in sandbox;
- validator scores on private data.

This makes the evaluated model much closer to the declared model.

### 11. Remote Attestation / Trusted Runtime

If Poker44 later uses attested runtime paths, the subnet can bind model identity more strongly
to actual execution.

### 12. Formal Compliance Policy

Over time, Poker44 can evolve from:

- open-source encouraged

to:

- open-source + disclosure recommended

to:

- open-source + disclosure + artifact verifiability required for top-tier eligibility.

## Practical Positioning

The correct public claim is:

- open source improves transparency and auditability;
- private rotating evaluation protects against simple memorization;
- future verification layers will tighten the gap between declared and executed models.

The incorrect public claim is:

- open source alone proves a miner is honest.

## Suggested Next Implementation Steps

1. Keep the current manifest support and require data-disclosure fields in miner docs.
2. Add a validator-side suspicion registry for missing manifests and manifest changes.
3. Add fingerprinting for served chunks so repeated exposure can be monitored.
4. Design a canary chunk generator and holdout policy.
5. Plan artifact-based verification as a later phase, not as the first move.
