import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


def _stringify_keys(tree):
    """Recursively convert dict keys to strings for Orbax/NNX compatibility."""
    if isinstance(tree, dict):
        return {str(k): _stringify_keys(v) for k, v in tree.items()}
    return tree


def _restore_keys(ref, merged):
    """Restore integer dict keys where the reference tree used them."""
    if not isinstance(ref, dict) or not isinstance(merged, dict):
        return merged
    ref_key_map = {str(k): k for k in ref}
    result = {}
    for key, value in merged.items():
        orig_key = ref_key_map.get(key, key)
        result[orig_key] = _restore_keys(ref.get(orig_key, {}), value)
    return result


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class NativeMEMWeightLoader(WeightLoader):
    """Loads base pi0.5 weights while keeping NativeMEM-only parameters initialized."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        str_params = _stringify_keys(params)
        merged_params = _merge_params(loaded_params, str_params, missing_regex=".*")
        return _restore_keys(params, merged_params)


def _remap_siglip_encoder_key(key: str, dest_prefix: str) -> str:
    suffix = key[len("PaliGemma/img/") :]
    return f"{dest_prefix}/{suffix}"


@dataclasses.dataclass(frozen=True)
class MemTokenizerWeightLoader(WeightLoader):
    """Loads pi0.5 weights into the mem_tokenizer student and frozen VLM."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        suffix_mlp_prefixes = ("action_in_proj/", "time_mlp_in/", "time_mlp_out/", "action_out_proj/")
        remapped = {}
        for key, value in flat_loaded.items():
            if key.startswith("PaliGemma/img/"):
                remapped[_remap_siglip_encoder_key(key, "encoder")] = value
                remapped[key] = value
            elif key.startswith(("PaliGemma/llm/", *suffix_mlp_prefixes)):
                remapped[key] = value

        remapped_params = flax.traverse_util.unflatten_dict(remapped, sep="/")
        str_params = _stringify_keys(params)
        merged_params = _merge_params(remapped_params, str_params, missing_regex=".*")
        return _restore_keys(params, merged_params)


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
        elif k.endswith("/value") and k[:-6] in flat_ref:
            # NNX checkpoints store Param leaves under ``value``, while
            # ``State.to_pure_dict()`` exposes the same leaves without it.
            ref_k = k[:-6]
            result[ref_k] = v.astype(flat_ref[ref_k].dtype) if v.dtype != flat_ref[ref_k].dtype else v

    load_num = len(result)
    if load_num == 0:
        raise ValueError(
            "No checkpoint weights matched the model parameter tree. "
            "Check that the checkpoint and model configurations are compatible."
        )
    logging.info("Loaded %d/%d weights from the checkpoint.", load_num, len(flat_ref))

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    logging.info("Merged %d/%d missing weights.", len(result) - load_num, len(flat_ref) - load_num)

    return flax.traverse_util.unflatten_dict(result, sep="/")
