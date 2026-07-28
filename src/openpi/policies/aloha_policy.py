import dataclasses
from typing import ClassVar, Literal

import einops
import numpy as np

from openpi import transforms

GripperType = Literal["trossen", "sim", "arx"]


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # Selects the gripper-space conversion. Only used when adapt_to_pi=True.
    #   "trossen": original Aloha pipeline (state in linear position, action in joint angle).
    #   "sim": state and action both in [0, 1] (0=close, 1=open).
    #   "arx": state and action both in [-pi, 0] (0=close, -pi=open).
    gripper_type: GripperType = "trossen"

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi, gripper_type=self.gripper_type)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi, gripper_type=self.gripper_type)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "memory" in data:
            inputs["memory"] = data["memory"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # Must match the gripper_type used for AlohaInputs so outputs round-trip to the original action space.
    gripper_type: GripperType = "trossen"

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi, gripper_type=self.gripper_type)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False, gripper_type: GripperType = "trossen") -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi, gripper_type=gripper_type)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False, gripper_type: GripperType = "trossen") -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_state_to_pi(state[[6, 13]], gripper_type)
    return state


def _encode_actions(
    actions: np.ndarray, *, adapt_to_pi: bool = False, gripper_type: GripperType = "trossen"
) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_action_from_pi(actions[:, [6, 13]], gripper_type)
    return actions


def _encode_actions_inv(
    actions: np.ndarray, *, adapt_to_pi: bool = False, gripper_type: GripperType = "trossen"
) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_action_to_pi(actions[:, [6, 13]], gripper_type)
    return actions


def _gripper_state_to_pi(value: np.ndarray, gripper_type: GripperType) -> np.ndarray:
    """Gripper state to pi0 internal space, normalized so 0=close and 1=open."""
    if gripper_type == "trossen":
        return _gripper_to_angular(value)
    if gripper_type == "sim":
        return value
    if gripper_type == "arx":
        return -value / np.pi
    raise ValueError(f"Unknown gripper_type: {gripper_type}")


def _gripper_action_to_pi(value: np.ndarray, gripper_type: GripperType) -> np.ndarray:
    """Gripper action command to pi0 internal space."""
    if gripper_type == "trossen":
        return _gripper_from_angular_inv(value)
    if gripper_type == "sim":
        return value
    if gripper_type == "arx":
        return -value / np.pi
    raise ValueError(f"Unknown gripper_type: {gripper_type}")


def _gripper_action_from_pi(value: np.ndarray, gripper_type: GripperType) -> np.ndarray:
    """Gripper action from pi0 internal space back to the original robot space."""
    if gripper_type == "trossen":
        return _gripper_from_angular(value)
    if gripper_type == "sim":
        return value
    if gripper_type == "arx":
        return -value * np.pi
    raise ValueError(f"Unknown gripper_type: {gripper_type}")
