# Open-Sourced Roadmap

Roadmap and implementation notes for Poker44's transition toward open-source miner models,
anti-leakage controls, and future compliance-based enforcement.

## Objective

Poker44 wants miner models to become open source for one core reason:

- reduce the risk that miners are effectively hardcoding or memorizing validator-private
  human hands instead of learning transferable bot-detection behavior.

Open source is not treated as full proof of honesty. It is the first layer of:

- transparency
- auditability
- training-data disclosure
- future compliance enforcement

## Current Architecture

Poker44 currently evaluates miners through remote inference:

1. validator builds mixed private chunks
2. validator sanitizes payloads
3. validator sends chunks to miners
4. miners return `risk_scores`
5. validator computes rewards
6. validator sets weights on-chain

At this stage, we are **not** changing this evaluation model.

## Why Open Source

Main concern:

- a miner could have access to validator-private human data
- a miner could overfit, memorize, or hardcode hands/chunks
- raw scoring alone would not be enough to distinguish genuine generalization from cheating

Therefore:

- `open_source=true` is useful
- but open source alone is not a sufficient defense

The correct positioning is:

- open source provides minimum transparency and traceability
- private rotating evaluation still protects against simple memorization
- future controls will tighten the gap between declared model and executed model

## What Has Already Been Implemented

### 1. Optional `model_manifest` in miner responses

Added to the current response contract without changing scoring.

Relevant files:

- `poker44/validator/synapse.py`
- `neurons/miner.py`
- `poker44/utils/model_manifest.py`

### 2. Reference miner publishes a manifest

The reference miner now publishes a manifest with:

- repo metadata
- model metadata
- training-data statement
- private-data attestation
- implementation hash

Relevant file:

- `neurons/miner.py`

### 3. Validator persists model metadata

The validator now records miner manifests by UID.

Relevant files:

- `neurons/validator.py`
- `poker44/validator/forward.py`

Generated registry:

- `model_manifests.json`

### 4. Anti-leakage tracking on validator side

Validator now also persists:

- `suspicion_registry.json`
- `served_chunk_registry.json`
- `compliance_registry.json`

Purpose:

- detect missing/incomplete manifests
- track repeated chunk exposure over time
- classify miners as `transparent` or `opaque`

Relevant files:

- `poker44/validator/integrity.py`
- `poker44/validator/forward.py`
- `neurons/validator.py`

### 5. Documentation added

Relevant docs:

- `docs/miner.md`
- `docs/validator.md`
- `docs/anti-leakage.md`
- `README.md`

## Current Compliance Standard

Miners are currently classified as:

- `transparent`
- `opaque`

Minimum fields required for `transparent` compliance:

- `open_source=true`
- `repo_url`
- `repo_commit`
- `model_name`
- `model_version`
- `training_data_statement`
- `private_data_attestation`

If these are missing, the miner is currently still evaluated and scored normally, but marked
as `opaque`.

## Important Current Constraint

At the moment:

- there is **no reward penalty**
- there is **no weight penalty**
- there is **no rejection from evaluation**
- the standard is **merged in the repo and pushed to GitHub**
- the standard is **not yet deployed on validator servers**

This phase is intentionally:

- logging
- tracking
- observability
- adoption pressure

Not punishment.

## Agreed Rollout Strategy

### Phase 1. Logging only

Current phase.

Behavior:

- keep all miners eligible
- log `transparent` / `opaque`
- persist compliance and suspicion registries
- let miners adapt gradually
- define the standard publicly before validator deployment
- communicate that repo/docs are ready before production rollout

Rationale:

- avoid breaking participation too early
- avoid penalizing miners before clear communication and transition time
- give miners time to prepare before the standard is live on validator infrastructure

### Phase 2. Social / operational pressure

Target window:

- approximately 2-3 weeks after communication starts

Behavior:

- continue without direct reward/weight penalty
- make `opaque` status more visible in logs/telemetry/docs
- communicate that future restrictions are coming

Possible additions:

- explicit compliance summary in validator logs
- separate reporting of transparent vs opaque miners
- more visible communication of compliant vs non-compliant miner status

### Phase 3. Soft restrictions

After the transition window.

Behavior:

- still not full punishment
- start restricting benefits or recognition for `opaque` miners

Examples:

- no compliant badge
- excluded from future featured lists / champion-style compliant paths
- lower operational trust tier

Important:

- this phase should be communicated publicly before activation

### Phase 4. Economic enforcement

Later phase, only after transition and communication.

Behavior:

- weights or rewards can begin to incorporate compliance policy
- `opaque` miners may become partially penalized
- later, non-compliant miners may become fully ineligible
- this future phase may also coincide with a higher miner emission share for compliant miners

Important:

- do not do this silently
- publish the policy before activating it
- give miners time to comply

## Anti-Leakage Position

Correct claim:

- open source improves transparency and auditability
- it does not by itself prove miners are honest

Real defense against leakage requires:

- private validator datasets
- rotating evaluation windows
- limited repeated exposure
- future canary chunks / hidden holdouts
- possibly artifact-based verification later

Incorrect claim:

- open source alone prevents cheating

## Next Recommended Implementation Phase

When resuming in another terminal, the next practical implementation target should be:

1. improve validator logging of compliance summary
2. optionally surface `transparent/opaque` more clearly in telemetry
3. design and implement canary chunks / holdout strategy
4. prepare future policy hooks for soft restrictions on `opaque` miners

## Current Deployment Status

Important for future sessions:

- phase 1 code is already implemented in the repository
- documentation is already updated in the repository
- commit is already pushed to GitHub remote
- validator servers are **not yet** updated to this version

This means:

- communication to miners should describe the standard as introduced/defined, not as already
  live on validator infrastructure
- miners can prepare now, before deployment
- any future live behavior depends on validator rollout

## Messaging Guidance For Miners

When communicating this update to miners, the messaging should emphasize:

- open-sourced model manifests
- transparency
- traceability
- structured ecosystem standards
- non-punitive rollout

Messaging should **not** claim:

- that validators are already enforcing the standard live, if deployment has not happened yet
- that there is already a reward or weight penalty
- that open source alone proves honesty

Preferred messaging shape:

- Poker44 is introducing an open-sourced model manifest standard
- the standard does not change current scoring or `set_weights`
- miners can already prepare by publishing manifests
- transparent/opaque classification exists in the design and code
- future compliance-based phases may later affect incentives

Useful public links:

- `https://github.com/Poker44/Poker44-subnet/blob/main/docs/miner.md`
- `https://github.com/Poker44/Poker44-subnet/blob/main/docs/validator.md`
- `https://github.com/Poker44/Poker44-subnet/blob/main/docs/anti-leakage.md`
- `https://github.com/Poker44/Poker44-subnet/blob/main/docs/opensourced_roadmap.md`

Preferred title style while validators are not yet deployed:

- `Poker44 miner update: Poker44 is introducing open-sourced model manifests`

Avoid title styles that imply live production rollout if validators are not yet upgraded.
