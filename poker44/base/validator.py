# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Poker44 Subnet

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt
from typing import List, Union
from traceback import print_exception
from poker44.base.neuron import BaseNeuron
from poker44.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)
from poker44.utils.config import add_validator_args
from poker44.validator.integrity import (
    persist_json_registry,
    remove_uid_from_compliance_registry,
    remove_uid_from_model_manifest_registry,
    remove_uid_from_suspicion_registry,
)
from poker44.validator.constants import BURN_EMISSIONS, BURN_FRACTION, UID_ZERO


def build_weight_vector_from_scores(scores: np.ndarray) -> np.ndarray:
    """Convert accumulated scores into on-chain weights while enforcing burn."""
    raw_scores = np.asarray(scores, dtype=np.float32).copy()
    raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
    raw_scores[raw_scores < 0] = 0.0

    if not BURN_EMISSIONS or raw_scores.size == 0 or UID_ZERO >= raw_scores.size:
        return raw_scores

    weight_vector = np.zeros_like(raw_scores)
    miner_scores = raw_scores.copy()
    miner_scores[UID_ZERO] = 0.0
    miner_total = float(np.sum(miner_scores))

    if miner_total <= 0.0:
        weight_vector[UID_ZERO] = 1.0
        return weight_vector

    weight_vector[UID_ZERO] = float(BURN_FRACTION)
    weight_vector += miner_scores / miner_total * float(1.0 - BURN_FRACTION)
    return weight_vector


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        self.dendrite = bt.Dendrite(wallet=self.wallet)

        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.axon = None

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.Axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon = getattr(self, "axon", None)
            if axon is not None:
                axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(
                str(print_exception(type(err), err, err.__traceback__))
            )

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self) -> bool:
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Convert accumulated scores into a weight vector while preserving the
        # intended burn ratio to UID 0 on-chain.
        raw_weights = build_weight_vector_from_scores(self.scores)

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)
        nonzero_uids = len(uint_uids)

        wait_for_inclusion = bool(self.config.neuron.wait_for_inclusion)
        wait_for_finalization = bool(self.config.neuron.wait_for_finalization)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
            version_key=self.spec_version,
        )
        if (
            result is False
            and "get_encrypted_commit() missing 1 required positional argument: 'hotkey'"
            in str(msg)
        ):
            bt.logging.warning(
                "Detected bittensor commit-reveal hotkey bug; retrying with local fallback."
            )
            try:
                result, msg = self._set_weights_commit_reveal_fallback(
                    uint_uids=uint_uids,
                    uint_weights=uint_weights,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
            except Exception as err:
                result, msg = False, str(err)
        if (
            result is False
            and "Call function 'SubtensorModule.commit_crv3_weights' not found"
            in str(msg)
        ):
            bt.logging.warning(
                "commit_crv3_weights is unavailable on the current RPC; falling back to classic set_weights."
            )
            result, msg = self._set_weights_classic_fallback(
                uint_uids=uint_uids,
                uint_weights=uint_weights,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        if result is True and (wait_for_inclusion or wait_for_finalization):
            bt.logging.info(f"set_weights confirmed on chain: {msg}")
        elif result is True:
            bt.logging.info(f"set_weights submitted to chain without confirmation: {msg}")
        else:
            bt.logging.error("set_weights failed", msg)
        write_snapshot = getattr(self, "_write_runtime_snapshot", None)
        if callable(write_snapshot):
            write_snapshot(
                status="running",
                extra={
                    "last_set_weights_success": bool(result),
                    "last_set_weights_message": str(msg),
                    "last_set_weights_wait_for_inclusion": wait_for_inclusion,
                    "last_set_weights_wait_for_finalization": wait_for_finalization,
                    "last_set_weights_nonzero_uids": nonzero_uids,
                },
            )
        wandb_helper = getattr(self, "wandb_helper", None)
        if wandb_helper is not None:
            wandb_helper.log_set_weights_result(
                success=bool(result),
                message=str(msg),
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        return bool(result)

    def _set_weights_commit_reveal_fallback(
        self,
        uint_uids: List[int],
        uint_weights: List[int],
        wait_for_inclusion: bool,
        wait_for_finalization: bool,
    ):
        """Fallback for bittensor 9.6.0 commit-reveal path, which omits `hotkey`."""
        from bittensor.core.extrinsics.commit_reveal import (
            _do_commit_reveal_v3,
            convert_and_normalize_weights_and_uids,
            get_encrypted_commit,
        )

        current_block = self.subtensor.get_current_block()
        subnet_hyperparameters = self.subtensor.get_subnet_hyperparameters(
            self.config.netuid, block=current_block
        )
        normalized_uids, normalized_weights = convert_and_normalize_weights_and_uids(
            uint_uids, uint_weights
        )
        commit_for_reveal, reveal_round = get_encrypted_commit(
            uids=normalized_uids,
            weights=normalized_weights,
            version_key=self.spec_version,
            tempo=subnet_hyperparameters.tempo,
            current_block=current_block,
            netuid=self.config.netuid,
            subnet_reveal_period_epochs=subnet_hyperparameters.commit_reveal_period,
            block_time=12.0,
            hotkey=self.wallet.hotkey.public_key,
        )
        return _do_commit_reveal_v3(
            subtensor=self.subtensor,
            wallet=self.wallet,
            netuid=self.config.netuid,
            commit=commit_for_reveal,
            reveal_round=reveal_round,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=8,
        )

    def _set_weights_classic_fallback(
        self,
        uint_uids: List[int],
        uint_weights: List[int],
        wait_for_inclusion: bool,
        wait_for_finalization: bool,
    ):
        """Fallback to classic set_weights when commit-reveal v3 is unavailable."""
        from bittensor.core.extrinsics.set_weights import set_weights_extrinsic

        return set_weights_extrinsic(
            subtensor=self.subtensor,
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            version_key=self.spec_version,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=8,
        )

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        prediction_buffer = getattr(self, "prediction_buffer", None)
        label_buffer = getattr(self, "label_buffer", None)
        model_manifest_registry = getattr(self, "model_manifest_registry", None)
        compliance_registry = getattr(self, "compliance_registry", None)
        suspicion_registry = getattr(self, "suspicion_registry", None)
        manifest_registry_changed = False
        compliance_registry_changed = False
        suspicion_registry_changed = False
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced
                if isinstance(prediction_buffer, dict):
                    prediction_buffer.pop(uid, None)
                if isinstance(label_buffer, dict):
                    label_buffer.pop(uid, None)
                if isinstance(model_manifest_registry, dict):
                    manifest_registry_changed = (
                        remove_uid_from_model_manifest_registry(model_manifest_registry, uid)
                        or manifest_registry_changed
                    )
                if isinstance(compliance_registry, dict):
                    compliance_registry_changed = (
                        remove_uid_from_compliance_registry(compliance_registry, uid)
                        or compliance_registry_changed
                    )
                if isinstance(suspicion_registry, dict):
                    suspicion_registry_changed = (
                        remove_uid_from_suspicion_registry(suspicion_registry, uid)
                        or suspicion_registry_changed
                    )

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        if len(self.hotkeys) > len(self.metagraph.hotkeys):
            removed_uids = range(len(self.metagraph.hotkeys), len(self.hotkeys))
            for uid in removed_uids:
                if isinstance(prediction_buffer, dict):
                    prediction_buffer.pop(uid, None)
                if isinstance(label_buffer, dict):
                    label_buffer.pop(uid, None)
                if isinstance(model_manifest_registry, dict):
                    manifest_registry_changed = (
                        remove_uid_from_model_manifest_registry(model_manifest_registry, uid)
                        or manifest_registry_changed
                    )
                if isinstance(compliance_registry, dict):
                    compliance_registry_changed = (
                        remove_uid_from_compliance_registry(compliance_registry, uid)
                        or compliance_registry_changed
                    )
                if isinstance(suspicion_registry, dict):
                    suspicion_registry_changed = (
                        remove_uid_from_suspicion_registry(suspicion_registry, uid)
                        or suspicion_registry_changed
                    )

        if manifest_registry_changed:
            persist_json_registry(
                getattr(self, "model_manifest_path", None),
                model_manifest_registry,
            )
        if compliance_registry_changed:
            persist_json_registry(
                getattr(self, "compliance_registry_path", None),
                compliance_registry,
            )
        if suspicion_registry_changed:
            persist_json_registry(
                getattr(self, "suspicion_registry_path", None),
                suspicion_registry,
            )

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Apply EMA only to the UIDs that were actually updated this cycle.
        # Non-sampled miners should retain their existing score instead of
        # decaying as if they had received an implicit zero reward.
        alpha: float = self.config.neuron.moving_average_alpha
        observed_scores = self.scores[uids_array]
        updated_scores = alpha * rewards + (1 - alpha) * observed_scores
        bt.logging.debug(f"Observed rewards: {rewards}")
        self.scores[uids_array] = updated_scores
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
            coverage_round_index=np.asarray(
                [int(getattr(self, "coverage_round_index", 0))], dtype=np.int64
            ),
            coverage_round_expected_uids=np.asarray(
                list(getattr(self, "coverage_round_expected_uids", [])), dtype=np.int64
            ),
            coverage_round_seen_uids=np.asarray(
                sorted(getattr(self, "coverage_round_seen_uids", set())), dtype=np.int64
            ),
            coverage_round_reward_sum_uids=np.asarray(
                sorted(getattr(self, "coverage_round_reward_sums", {}).keys()), dtype=np.int64
            ),
            coverage_round_reward_sum_values=np.asarray(
                [
                    float(getattr(self, "coverage_round_reward_sums", {}).get(uid, 0.0))
                    for uid in sorted(getattr(self, "coverage_round_reward_sums", {}).keys())
                ],
                dtype=np.float32,
            ),
            coverage_round_reward_count_uids=np.asarray(
                sorted(getattr(self, "coverage_round_reward_counts", {}).keys()), dtype=np.int64
            ),
            coverage_round_reward_count_values=np.asarray(
                [
                    int(getattr(self, "coverage_round_reward_counts", {}).get(uid, 0))
                    for uid in sorted(getattr(self, "coverage_round_reward_counts", {}).keys())
                ],
                dtype=np.int64,
            ),
            coverage_round_pending_set_weights=np.asarray(
                [int(bool(getattr(self, "coverage_round_pending_set_weights", False)))],
                dtype=np.int64,
            ),
            coverage_round_completed_at_step=np.asarray(
                [
                    int(getattr(self, "coverage_round_completed_at_step", -1))
                    if getattr(self, "coverage_round_completed_at_step", None) is not None
                    else -1
                ],
                dtype=np.int64,
            ),
        )
        write_snapshot = getattr(self, "_write_runtime_snapshot", None)
        if callable(write_snapshot):
            write_snapshot(
                status="running",
                extra={
                    "step": int(self.step),
                    "score_slots": int(len(self.scores)),
                    "nonzero_scores": int(np.count_nonzero(self.scores)),
                },
            )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        state = np.load(self.config.neuron.full_path + "/state.npz")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
        if "coverage_round_index" in state:
            self.coverage_round_index = int(state["coverage_round_index"][0])
            self.coverage_round_expected_uids = [
                int(uid) for uid in state["coverage_round_expected_uids"].tolist()
            ]
            self.coverage_round_seen_uids = {
                int(uid) for uid in state["coverage_round_seen_uids"].tolist()
            }
            self.coverage_round_reward_sums = {
                int(uid): float(value)
                for uid, value in zip(
                    state["coverage_round_reward_sum_uids"].tolist(),
                    state["coverage_round_reward_sum_values"].tolist(),
                )
            }
            self.coverage_round_reward_counts = {
                int(uid): int(value)
                for uid, value in zip(
                    state["coverage_round_reward_count_uids"].tolist(),
                    state["coverage_round_reward_count_values"].tolist(),
                )
            }
            self.coverage_round_pending_set_weights = bool(
                int(state["coverage_round_pending_set_weights"][0])
            )
            completed_at_step = int(state["coverage_round_completed_at_step"][0])
            self.coverage_round_completed_at_step = (
                completed_at_step if completed_at_step >= 0 else None
            )
