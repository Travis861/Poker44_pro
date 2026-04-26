"""Asynchronous forward loop for the Poker44 validator."""
## poker44/validator/forward.py

from __future__ import annotations

import asyncio
import math
import os
import time
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import bittensor as bt
import numpy as np

from poker44.score.scoring import reward
from poker44.utils.model_manifest import manifest_digest, normalize_model_manifest
from poker44.validator.integrity import (
    chunk_fingerprint,
    evaluate_manifest_compliance,
    evaluate_manifest_suspicion,
    normalize_uid_key_registry,
    persist_json_registry,
    record_served_chunks,
    update_compliance_registry,
    update_suspicion_registry,
)
from poker44.validator.synapse import DetectionSynapse
from poker44.validator.sanitization import sanitize_hand_for_miner

from poker44.validator.constants import (
    BURN_EMISSIONS,
    BURN_FRACTION,
    KEEP_FRACTION,
    UID_ZERO,
    WINNER_TAKE_ALL,
)


async def forward(validator) -> None:
    """Entry point invoked by :class:`neurons.validator.Validator`."""
    try:
        await _run_forward_cycle(validator)
    except Exception:
        wandb_helper = getattr(validator, "wandb_helper", None)
        if wandb_helper is not None:
            wandb_helper.log_error(
                "forward_cycle_unexpected",
                traceback.format_exc(),
            )
        bt.logging.error(f"Unexpected error in forward cycle:\n{traceback.format_exc()}")


async def _run_forward_cycle(validator) -> None:
    validator.forward_count = getattr(validator, "forward_count", 0) + 1
    bt.logging.info(f"[Forward #{validator.forward_count}] start")
    wandb_helper = getattr(validator, "wandb_helper", None)

    if hasattr(validator.provider, "refresh_if_due"):
        validator.provider.refresh_if_due()

    # Fetch all configured chunks from the stable dataset snapshot.
    chunk_limit = int(getattr(validator, "chunk_batch_size", 80))
    batches = validator.provider.fetch_hand_batch(limit=chunk_limit)
    if not batches:
        bt.logging.info("No hands fetched from dataset; sleeping.")
        if wandb_helper is not None:
            wandb_helper.log_forward_summary(
                forward_count=validator.forward_count,
                chunk_count=0,
                total_hands=0,
                miner_count=0,
                responded_count=0,
                successful_miners=0,
                dataset_hash=getattr(validator.provider, "dataset_hash", ""),
                dataset_stats=getattr(validator.provider, "stats", {}),
                extra={"forward/status": "no_batches"},
            )
        await asyncio.sleep(validator.poll_interval)
        return

    provider_stats = getattr(validator.provider, "stats", {}) or {}
    current_window_id = provider_stats.get("window_id")
    try:
        resolved_window_id = (
            int(current_window_id) if current_window_id is not None else None
        )
    except (TypeError, ValueError):
        resolved_window_id = None

    previous_window_id = getattr(validator, "current_eval_window_id", None)
    if (
        getattr(validator, "sync_reset_buffers_on_window_change", False)
        and resolved_window_id is not None
        and previous_window_id is not None
        and previous_window_id != resolved_window_id
    ):
        validator.prediction_buffer = {}
        validator.label_buffer = {}
        bt.logging.info(
            f"Eval window changed ({previous_window_id} -> {resolved_window_id}); "
            "cleared local buffers."
        )
    validator.current_eval_window_id = resolved_window_id
    bt.logging.info(
        "Using evaluation snapshot | "
        f"window_id={resolved_window_id} "
        f"dataset_hash={getattr(validator.provider, 'dataset_hash', '')[:12]}"
    )
    
    eligible_miner_uids, miner_uids, axons = _get_candidate_miners(validator)
    validator.ensure_coverage_round(
        eligible_miner_uids,
        reason="forward cycle bootstrap",
    )

    if getattr(validator, "coverage_round_pending_set_weights", False):
        bt.logging.info(
            f"Coverage round #{getattr(validator, 'coverage_round_index', 0)} is complete; "
            "waiting for the next permitted set_weights window before querying more miners."
        )
        await asyncio.sleep(validator.poll_interval)
        return

    responses: Dict[int, List[float]] = {uid: [] for uid in miner_uids}
    cycle_predictions: Dict[int, List[float]] = {uid: [] for uid in miner_uids}
    cycle_labels: Dict[int, List[int]] = {uid: [] for uid in miner_uids}

    if not miner_uids:
        bt.logging.info("No eligible miner UIDs available for this cycle.")
        if wandb_helper is not None:
            wandb_helper.log_forward_summary(
                forward_count=validator.forward_count,
                chunk_count=len(batches),
                total_hands=sum(len(batch.hands) for batch in batches),
                miner_count=0,
                responded_count=0,
                successful_miners=0,
                dataset_hash=getattr(validator.provider, "dataset_hash", ""),
                dataset_stats=getattr(validator.provider, "stats", {}),
                extra={"forward/status": "no_eligible_miners"},
            )
        await asyncio.sleep(validator.poll_interval)
        return
    
    # Prepare chunks and labels
    chunks = []  # List of batches (each batch is a list of hand dicts)
    batch_labels = []  # One label per batch
    
    for batch in batches:
        # Convert HandHistory objects to dicts
        chunk_dicts = []
        for hand in batch.hands:
            hand_payload: Dict[str, Any]
            if isinstance(hand, dict):
                hand_payload = hand
            else:
                # Assume hand has a to_payload() or to_dict() method
                try:
                    hand_payload = hand.to_payload()
                except AttributeError:
                    # Fallback: convert dataclass to dict
                    import dataclasses
                    if dataclasses.is_dataclass(hand):
                        hand_payload = dataclasses.asdict(hand)
                    else:
                        hand_payload = hand.__dict__

            chunk_dicts.append(sanitize_hand_for_miner(hand_payload))
        
        chunks.append(chunk_dicts)
        
        # batch.is_human is False for bots, True for humans
        # We need: 1=bot, 0=human
        batch_label = 0 if batch.is_human else 1
        batch_labels.append(batch_label)
    
    bt.logging.info(f"Processing {len(chunks)} chunks with labels: {batch_labels} (1=bot, 0=human)")
    bt.logging.info(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    _record_served_chunk_fingerprints(
        validator,
        chunks=chunks,
        dataset_hash=getattr(validator.provider, "dataset_hash", ""),
    )
    if wandb_helper is not None:
        wandb_helper.log_dataset_state(
            dataset_hash=getattr(validator.provider, "dataset_hash", ""),
            stats=provider_stats,
        )
    
    # Create synapse with all chunks (now as list of dicts)
    synapse = DetectionSynapse(chunks=chunks)
    
    # Get timeout from config
    timeout = 20
    if hasattr(validator.config, "neuron") and hasattr(validator.config.neuron, "timeout"):
        try:
            timeout = float(validator.config.neuron.timeout)
        except (ValueError, TypeError):
            timeout = 20
    
    total_hands = sum(len(chunk) for chunk in chunks)
    bt.logging.info(f"Querying {len(axons)} miners with {len(chunks)} chunks ({total_hands} total hands)...")
    
    synapse_responses = await _dendrite_with_retries(
        validator.dendrite,
        axons=axons,
        synapse=synapse,
        timeout=timeout,
        attempts=3,
    )
    bt.logging.info(f"Received {len(synapse_responses)} responses from miners")
    
    for uid, resp in zip(miner_uids, synapse_responses):
        if resp is None:
            bt.logging.debug(f"Miner {uid} returned None response")
            continue

        _record_model_manifest(
            validator,
            uid,
            getattr(resp, "model_manifest", None),
            dataset_hash=getattr(validator.provider, "dataset_hash", ""),
        )
            
        scores = getattr(resp, "risk_scores", None)
        if scores is None:
            bt.logging.debug(f"Miner {uid} returned no risk_scores")
            continue
            
        try:
            scores_f = [float(s) for s in scores]
            
            # Miners should return one score per chunk
            if len(scores_f) != len(chunks):
                bt.logging.warning(
                    f"Miner {uid} returned {len(scores_f)} scores but expected {len(chunks)} (one per chunk)"
                )
                # Continue anyway, use what we have
                min_len = min(len(scores_f), len(chunks))
                scores_f = scores_f[:min_len]
                effective_labels = batch_labels[:min_len]
            else:
                effective_labels = batch_labels
            
            responses[uid].extend(scores_f)
            cycle_predictions[uid].extend(scores_f)
            cycle_labels[uid].extend(effective_labels)
            
            # Store predictions and labels (one per chunk)
            if not hasattr(validator, "prediction_buffer"):
                validator.prediction_buffer = {}
            if not hasattr(validator, "label_buffer"):
                validator.label_buffer = {}
            
            max_buffer_size = max(1, int(getattr(validator, "reward_window", len(scores_f) or 1)))
            prediction_history = validator.prediction_buffer.setdefault(uid, [])
            label_history = validator.label_buffer.setdefault(uid, [])
            prediction_history.extend(scores_f)
            label_history.extend(effective_labels)
            if len(prediction_history) > max_buffer_size:
                del prediction_history[:-max_buffer_size]
            if len(label_history) > max_buffer_size:
                del label_history[:-max_buffer_size]
            
            bt.logging.info(f"Miner {uid} scored {len(scores_f)} chunks successfully")
        except Exception as e:
            bt.logging.warning(f"Error processing response from miner {uid}: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            continue
    
    if not any(responses.values()):
        bt.logging.info("No miner responses this cycle.")
        if wandb_helper is not None:
            wandb_helper.log_forward_summary(
                forward_count=validator.forward_count,
                chunk_count=len(chunks),
                total_hands=total_hands,
                miner_count=len(axons),
                responded_count=len(synapse_responses),
                successful_miners=0,
                dataset_hash=getattr(validator.provider, "dataset_hash", ""),
                dataset_stats=getattr(validator.provider, "stats", {}),
                extra={
                    "forward/status": "no_valid_scores",
                    "forward/human_chunk_count": sum(1 for label in batch_labels if label == 0),
                    "forward/bot_chunk_count": sum(1 for label in batch_labels if label == 1),
                },
            )
        await asyncio.sleep(validator.poll_interval)
        return
    
    if getattr(validator, "sync_direct_score_update", False):
        rewards_array, metrics = _compute_cycle_rewards(
            miner_uids,
            cycle_predictions=cycle_predictions,
            cycle_labels=cycle_labels,
        )
    else:
        rewards_array, metrics = _compute_windowed_rewards(validator, miner_uids)
    reward_map = dict(zip(miner_uids, rewards_array.tolist()))
    metrics_map = {uid: metric for uid, metric in zip(miner_uids, metrics)}
    validator.record_round_cycle(sampled_uids=miner_uids, reward_map=reward_map)
    bt.logging.info(f"Reward map by UID: {reward_map}")
    bt.logging.info(f"Reward metrics by UID: {metrics_map}")
    winner_uids, winner_rewards = _select_weight_targets(reward_map)

    if getattr(validator, "sync_direct_score_update", False):
        _apply_synced_scores(validator, winner_uids, winner_rewards)
    else:
        validator.update_scores(winner_rewards, winner_uids)
    if wandb_helper is not None:
        successful_miners = sum(1 for scores in responses.values() if scores)
        wandb_helper.log_forward_summary(
            forward_count=validator.forward_count,
            chunk_count=len(chunks),
            total_hands=total_hands,
            miner_count=len(axons),
            responded_count=len(synapse_responses),
            successful_miners=successful_miners,
            dataset_hash=getattr(validator.provider, "dataset_hash", ""),
            dataset_stats=getattr(validator.provider, "stats", {}),
            extra={
                "forward/status": "ok",
                "forward/human_chunk_count": sum(1 for label in batch_labels if label == 0),
                "forward/bot_chunk_count": sum(1 for label in batch_labels if label == 1),
                "forward/window_id": resolved_window_id if resolved_window_id is not None else -1,
                "forward/miner_uid_count": len(miner_uids),
            },
        )
        wandb_helper.log_reward_summary(
            reward_map=reward_map,
            metrics_map=metrics_map,
            winner_uids=[int(uid) for uid in winner_uids],
            winner_rewards=[float(weight) for weight in winner_rewards],
        )
    bt.logging.info(f"Rewards issued for {len(winner_rewards)} UID(s).")
    bt.logging.info(
        f"[Forward #{validator.forward_count}] complete. Sleeping {validator.poll_interval}s before next tick.",
    )
    await asyncio.sleep(validator.poll_interval)


def _record_model_manifest(
    validator,
    uid: int,
    manifest: Dict[str, Any] | None,
    *,
    dataset_hash: str,
) -> None:
    normalized = normalize_model_manifest(manifest)
    suspicion_reasons = evaluate_manifest_suspicion(normalized if normalized else None)
    _record_suspicion(
        validator,
        uid,
        reasons=suspicion_reasons,
        dataset_hash=dataset_hash,
    )
    _record_compliance(
        validator,
        uid,
        manifest=normalized if normalized else None,
        dataset_hash=dataset_hash,
    )

    if not normalized:
        bt.logging.debug(f"Miner {uid} did not provide a model manifest.")
        return

    digest = manifest_digest(normalized)
    registry = getattr(validator, "model_manifest_registry", None)
    if registry is None:
        registry = {}
        validator.model_manifest_registry = registry

    registry_key = str(int(uid))
    previous = registry.get(registry_key)
    previous_digest = previous.get("manifest_digest") if previous else None
    if previous_digest == digest:
        return

    entry = {
        "uid": int(uid),
        "manifest_digest": digest,
        "model_manifest": normalized,
    }
    registry[registry_key] = entry

    bt.logging.info(
        f"Miner {uid} manifest updated | "
        f"open_source={normalized.get('open_source')} "
        f"model={normalized.get('model_name', '')} "
        f"version={normalized.get('model_version', '')} "
        f"repo={normalized.get('repo_url', '')} "
        f"commit={normalized.get('repo_commit', '')}"
    )
    _persist_model_manifest_registry(getattr(validator, "model_manifest_path", None), registry)


def _persist_model_manifest_registry(
    path: str | Path | None,
    registry: Dict[Any, Dict[str, Any]],
) -> None:
    # JSON round-tripping turns top-level dict keys into strings. Normalize on
    # every persist so reloaded registries never mix int and str UIDs.
    normalized_registry = normalize_uid_key_registry(registry)
    registry.clear()
    registry.update(normalized_registry)
    persist_json_registry(path, normalized_registry)


def _record_served_chunk_fingerprints(validator, *, chunks: List[List[dict]], dataset_hash: str) -> None:
    registry = getattr(validator, "served_chunk_registry", None)
    if registry is None:
        registry = {"chunk_index": {}, "recent_cycles": [], "summary": {}}
        validator.served_chunk_registry = registry

    chunk_hashes = [chunk_fingerprint(chunk) for chunk in chunks]
    summary = record_served_chunks(
        registry,
        chunk_hashes=chunk_hashes,
        forward_count=int(getattr(validator, "forward_count", 0)),
        dataset_hash=dataset_hash,
    )
    persist_json_registry(getattr(validator, "served_chunk_registry_path", None), registry)

    if summary["repeated_count"] > 0:
        bt.logging.warning(
            f"Forward #{getattr(validator, 'forward_count', 0)} reused "
            f"{summary['repeated_count']} chunk fingerprints; "
            f"{summary['unique_count']} unique chunk fingerprints tracked so far."
        )


def _record_suspicion(
    validator,
    uid: int,
    *,
    reasons: List[str],
    dataset_hash: str,
) -> None:
    registry = getattr(validator, "suspicion_registry", None)
    if registry is None:
        registry = {"miners": {}, "summary": {}}
        validator.suspicion_registry = registry

    event = update_suspicion_registry(
        registry,
        uid=int(uid),
        reasons=reasons,
        forward_count=int(getattr(validator, "forward_count", 0)),
        dataset_hash=dataset_hash,
    )
    if event is None:
        return

    bt.logging.warning(f"Miner {uid} anti-leakage suspicion flags: {', '.join(reasons)}")
    persist_json_registry(getattr(validator, "suspicion_registry_path", None), registry)


def _record_compliance(
    validator,
    uid: int,
    *,
    manifest: Dict[str, Any] | None,
    dataset_hash: str,
) -> None:
    registry = getattr(validator, "compliance_registry", None)
    if registry is None:
        registry = {"miners": {}, "summary": {}}
        validator.compliance_registry = registry

    compliance = evaluate_manifest_compliance(manifest)
    digest = manifest_digest(manifest or {})
    entry = update_compliance_registry(
        registry,
        uid=int(uid),
        compliance=compliance,
        manifest_digest=digest,
        forward_count=int(getattr(validator, "forward_count", 0)),
        dataset_hash=dataset_hash,
    )
    persist_json_registry(getattr(validator, "compliance_registry_path", None), registry)

    if entry.get("status_changed"):
        bt.logging.info(
            f"Miner {uid} compliance status changed to {entry['status']} "
            f"(missing_fields={entry['missing_fields']})"
        )


def _get_candidate_miners(validator) -> Tuple[List[int], List[int], List]:
    miner_uids: List[int] = []
    axons: List = []
    target_uids_env = os.getenv("POKER44_TARGET_MINER_UIDS", "").strip()
    miners_per_cycle_env = os.getenv("POKER44_MINERS_PER_CYCLE", "16").strip()
    miners_per_cycle = 0
    target_uids = None
    if target_uids_env:
        try:
            target_uids = {
                int(uid.strip())
                for uid in target_uids_env.split(",")
                if uid.strip() != ""
            }
            bt.logging.info(f"Restricting miner queries to target UIDs: {sorted(target_uids)}")
        except ValueError:
            bt.logging.warning(
                f"Invalid POKER44_TARGET_MINER_UIDS={target_uids_env!r}; ignoring filter."
            )
            target_uids = None
    try:
        miners_per_cycle = int(miners_per_cycle_env)
    except ValueError:
        bt.logging.warning(
            f"Invalid POKER44_MINERS_PER_CYCLE={miners_per_cycle_env!r}; defaulting to 16 miners per cycle."
        )
        miners_per_cycle = 16

    for uid, axon in enumerate(validator.metagraph.axons):
        if uid == UID_ZERO:
            continue
        if target_uids is not None and uid not in target_uids:
            continue
        if bool(validator.metagraph.validator_permit[uid]):
            continue
        ip = str(getattr(axon, "ip", "") or "")
        port = int(getattr(axon, "port", 0) or 0)
        if ip in {"", "0.0.0.0", "::", "[::]"} or port <= 0:
            continue
        miner_uids.append(uid)
        axons.append(axon)

    ordered = sorted(zip(miner_uids, axons), key=lambda item: item[0])
    miner_uids = [uid for uid, _ in ordered]
    axons = [axon for _, axon in ordered]

    eligible_miner_uids = list(miner_uids)

    if getattr(validator, "sync_all_miners", False):
        bt.logging.info("Synchronized validator mode: querying all eligible miners.")
    elif target_uids is None and miners_per_cycle > 0 and len(miner_uids) > miners_per_cycle:
        # Rotate deterministically through the eligible set so coverage expands over time
        # without blasting every miner on each cycle. We key rotation to both the
        # shared evaluation window and a synchronized subwindow index derived from
        # wall clock / poll interval, so a validator does not stay pinned to the
        # same subset for the entire window.
        shared_window_id = int(getattr(validator, "current_eval_window_id", 0) or 0)
        validator_uid = int(getattr(validator, "uid", 0) or 0)
        stagger = validator_uid % len(miner_uids)
        poll_interval = max(1, int(getattr(validator, "poll_interval", 300) or 300))
        subwindow_id = math.floor(time.time() / poll_interval)
        expected_uids = set(getattr(validator, "coverage_round_expected_uids", []) or [])
        seen_uids = set(getattr(validator, "coverage_round_seen_uids", set()) or set())
        coverage_round_active = bool(expected_uids) and not bool(
            getattr(validator, "coverage_round_pending_set_weights", False)
        )

        pool = list(zip(miner_uids, axons))
        unseen_pool = [
            (uid, axon)
            for uid, axon in pool
            if uid in expected_uids and uid not in seen_uids
        ]

        if coverage_round_active and unseen_pool:
            offset = (
                (shared_window_id * miners_per_cycle)
                + (subwindow_id * miners_per_cycle)
                + stagger
            ) % len(unseen_pool)
            rotated = unseen_pool[offset:] + unseen_pool[:offset]
            selected = rotated[: min(miners_per_cycle, len(rotated))]
            bt.logging.info(
                f"Sampling {len(selected)} unseen miners this cycle from {len(unseen_pool)} "
                f"remaining unseen / {len(pool)} eligible miners "
                f"(window rotation offset={offset}, validator_stagger={stagger}, subwindow_id={subwindow_id})."
            )
        else:
            offset = (
                (shared_window_id * miners_per_cycle)
                + (subwindow_id * miners_per_cycle)
                + stagger
            ) % len(miner_uids)
            rotated = pool[offset:] + pool[:offset]
            selected = rotated[:miners_per_cycle]
            bt.logging.info(
                f"Sampling {miners_per_cycle} miners this cycle from {len(rotated)} eligible miners "
                f"(window rotation offset={offset}, validator_stagger={stagger}, subwindow_id={subwindow_id})."
            )

        miner_uids = [uid for uid, _ in selected]
        axons = [axon for _, axon in selected]

    bt.logging.info(f"Eligible miners this cycle: {miner_uids}")
    return eligible_miner_uids, miner_uids, axons


def _compute_windowed_rewards(validator, miner_uids: List[int]) -> tuple[np.ndarray, list]:
    window = getattr(validator, "reward_window", 20)
    rewards: List[float] = []
    metrics: List[dict] = []

    for uid in miner_uids:
        pred_buf = validator.prediction_buffer.get(uid, [])
        label_buf = validator.label_buffer.get(uid, [])

        if len(pred_buf) < window or len(label_buf) < window:
            rewards.append(0.0)
            metrics.append(
                {
                    "fpr": 1.0,
                    "bot_recall": 0.0,
                    "ap_score": 0.0,
                    "human_safety_penalty": 0.0,
                    "base_score": 0.0,
                    "reward": 0.0,
                }
            )
            continue

        preds_window = np.asarray(pred_buf[-window:], dtype=float)
        labels_window = np.asarray(label_buf[-window:], dtype=bool)
        rew, metric = reward(preds_window, labels_window)
        rewards.append(rew)
        metrics.append(metric)

    rewards_array = np.asarray(rewards, dtype=np.float32)
    
    return rewards_array, metrics


def _compute_cycle_rewards(
    miner_uids: List[int],
    *,
    cycle_predictions: Dict[int, List[float]],
    cycle_labels: Dict[int, List[int]],
) -> tuple[np.ndarray, list]:
    rewards: List[float] = []
    metrics: List[dict] = []

    for uid in miner_uids:
        preds = cycle_predictions.get(uid, [])
        labels = cycle_labels.get(uid, [])

        if not preds or not labels or len(preds) != len(labels):
            rewards.append(0.0)
            metrics.append(
                {
                    "fpr": 1.0,
                    "bot_recall": 0.0,
                    "ap_score": 0.0,
                    "human_safety_penalty": 0.0,
                    "base_score": 0.0,
                    "reward": 0.0,
                }
            )
            continue

        preds_arr = np.asarray(preds, dtype=float)
        labels_arr = np.asarray(labels, dtype=bool)
        rew, metric = reward(preds_arr, labels_arr)
        rewards.append(rew)
        metrics.append(metric)

    return np.asarray(rewards, dtype=np.float32), metrics


def _apply_synced_scores(
    validator,
    winner_uids: List[int],
    winner_rewards: np.ndarray,
) -> None:
    validator.scores = np.zeros_like(validator.scores)
    for uid, reward_value in zip(winner_uids, winner_rewards.tolist()):
        validator.scores[int(uid)] = float(reward_value)
    bt.logging.info(
        "Applied direct synchronized score vector for current evaluation window "
        f"(weight_targets={len(winner_uids)})."
    )


def _select_weight_targets(reward_map: Dict[int, float]) -> tuple[List[int], np.ndarray]:
    if not reward_map:
        bt.logging.info("No eligible rewards computed; assigning 100%% to UID 0.")
        return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

    sorted_rewards = sorted(reward_map.items(), key=lambda item: (-item[1], item[0]))
    winner_uid, winner_reward = sorted_rewards[0]

    if not WINNER_TAKE_ALL:
        positive = [(uid, max(0.0, float(reward))) for uid, reward in sorted_rewards]
        positive = [(uid, reward) for uid, reward in positive if reward > 0.0]

        if not positive:
            bt.logging.info(
                "No miner achieved positive reward; assigning 100%% to UID 0."
            )
            return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

        total_positive = float(sum(reward for _, reward in positive))
        if total_positive <= 0.0:
            bt.logging.info(
                "Positive-reward sum is zero; assigning 100%% to UID 0."
            )
            return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

        norm = [(uid, reward / total_positive) for uid, reward in positive]

        if BURN_EMISSIONS:
            uids = [UID_ZERO] + [uid for uid, _ in norm]
            rewards = [BURN_FRACTION] + [KEEP_FRACTION * frac for _, frac in norm]
            bt.logging.info(
                f"Proportional mode + burn: UID 0 gets {BURN_FRACTION * 100:.2f}%, "
                f"{KEEP_FRACTION * 100:.2f}% split across {len(norm)} miner(s)."
            )
            return uids, np.asarray(rewards, dtype=np.float32)

        uids = [uid for uid, _ in norm]
        rewards = [frac for _, frac in norm]
        bt.logging.info(f"Proportional mode: 100% split across {len(norm)} miner(s).")
        return uids, np.asarray(rewards, dtype=np.float32)

    if winner_reward <= 0.0:
        bt.logging.info("No miner achieved positive reward; assigning 100%% to UID 0.")
        return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

    if BURN_EMISSIONS:
        bt.logging.info(
            f"Winner-take-all burn enabled: UID 0 gets {BURN_FRACTION * 100:.2f}%, "
            f"winner UID {winner_uid} gets {KEEP_FRACTION * 100:.2f}%."
        )
        return [UID_ZERO, winner_uid], np.asarray(
            [BURN_FRACTION, KEEP_FRACTION], dtype=np.float32
        )

    bt.logging.info(f"Winner-take-all enabled: winner UID {winner_uid} gets 100%.")
    return [winner_uid], np.asarray([1.0], dtype=np.float32)

async def _dendrite_with_retries(
    dendrite: bt.dendrite,
    *,
    axons: Sequence,
    synapse: DetectionSynapse,
    timeout: float,
    attempts: int = 3,
):
    """
    Simple retry loop around dendrite calls to avoid transient failures.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return await dendrite(
                axons=axons,
                synapse=synapse,
                timeout=timeout,
            )
        except Exception as exc:
            last_exc = exc
            bt.logging.warning(f"dendrite attempt {attempt}/{attempts} failed: {exc}")
            await asyncio.sleep(0.5)
    bt.logging.error(f"dendrite retries exhausted: {last_exc}")
    return [None] * len(axons)
