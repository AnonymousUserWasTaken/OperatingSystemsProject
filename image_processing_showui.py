# coding=utf-8
# Copyright 2024 The Show Lab, National University of Singapore
# and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for ShowUI, inherited from Qwen2-VL."""

import math
from typing import Dict, List, Optional, Union

import PIL
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.segmentation import mark_boundaries

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    VideoInput,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


# Copied from transformers.models.llava_next_video.image_processing_llava_next_video.make_batched_videos
def make_batched_videos(videos) -> List[VideoInput]:
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], Image.Image):
            return [videos]
        elif len(videos[0].shape) == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]

    raise ValueError(f"Could not make batched video from {videos}")


# Implement Union-Find operator for constructing ui patches
class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[py] = px


# NOTE: This helper is unchanged; we now call it with a *smaller* runtime max_pixels.
def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 320,  # original default, but we override in _preprocess
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class ShowUIImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ShowUI image processor that inherited from Qwen2-VL,
    enabling UI-guided visual token selection.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to CLIP mean):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*, defaults to CLIP std):
            Standard deviation to use if normalizing the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 320`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder (NOTE: we effectively
            treat this as 1 in the new per-frame patching path).
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to LLM encoder.
    """

    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 320,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        self.do_convert_rgb = do_convert_rgb

        # 🔧 Runtime safety clamp: we never allow more than 224x224 pixels per frame
        # even if config.max_pixels is larger. This massively reduces token count
        # and RAM usage.
        self.runtime_max_pixels = min(self.max_pixels, 224 * 224)

    def rerank_values(self, arr):
        mapping = {}
        new_arr = np.empty_like(arr)
        next_value = 0

        for idx, x in enumerate(arr):
            if x not in mapping:
                mapping[x] = next_value
                next_value += 1
            new_arr[idx] = mapping[x]
        return new_arr

    def _build_uigraph(
        self,
        patches,
        grid_t,
        grid_h,
        grid_w,
        grid_h_half,
        grid_w_half,
        uigraph_threshold,
        channel,
    ):
        num_patches = grid_t * grid_h_half * grid_w_half
        uf = UnionFind(num_patches)

        def idx(t, i, j):
            return t * grid_h_half * grid_w_half + i * grid_w_half + j

        # Compare adjacent patches based on the threshold
        for t in range(grid_t):
            for i in range(grid_h_half):
                for j in range(grid_w_half):
                    current_idx = idx(t, i, j)
                    current_patch = patches[t, i, j, :, :, :, :,]  # (channel, temporal_patch_size, patch_size, patch_size)

                    # Compare with right neighbor
                    if j + 1 < grid_w_half:
                        right_patch = patches[t, i, j + 1, :, :, :, :,]
                        diff = np.linalg.norm(current_patch - right_patch)
                        if diff < uigraph_threshold:
                            uf.union(current_idx, idx(t, i, j + 1))

                    # Compare with bottom neighbor
                    if i + 1 < grid_h_half:
                        bottom_patch = patches[t, i + 1, j, :, :, :, :,]
                        diff = np.linalg.norm(current_patch - bottom_patch)
                        if diff < uigraph_threshold:
                            uf.union(current_idx, idx(t, i + 1, j))

        uigraph_assign_flat = np.array([uf.find(x) for x in range(num_patches)])
        le = LabelEncoder()
        uigraph_assign_flat = le.fit_transform(uigraph_assign_flat)
        uigraph_assign = uigraph_assign_flat.reshape((grid_t, grid_h_half, grid_w_half))
        return uigraph_assign

    def _vis_uigraph(self, uigraph_assign, image_size, patch_size, image):
        resized_height, resized_width = image_size[0]
        uigraph_assign = uigraph_assign[0]

        upscaled_uigraph_assign = np.repeat(np.repeat(uigraph_assign, patch_size, axis=0), patch_size, axis=1)
        upscaled_uigraph_assign = upscaled_uigraph_assign[:resized_height, :resized_width]

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if image.shape[0] in [1, 3]:  # Assuming grayscale or RGB image
            image = image.transpose(1, 2, 0)
        elif image.shape[2] in [1, 3]:
            pass
        else:
            raise ValueError("Unexpected image shape: {}".format(image.shape))

        boundaries_image = mark_boundaries(image, upscaled_uigraph_assign, color=(1, 0.4, 0.4))
        boundaries_image = (boundaries_image * 255).astype(np.uint8)
        return Image.fromarray(boundaries_image)

    # ============================================================
    # In-place, tiled rescale + normalize to avoid float64 temps
    # ============================================================
    def _rescale_and_normalize_inplace(
        self,
        image: np.ndarray,
        *,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        input_data_format: ChannelDimension,
        tile_h: int = 256,
    ) -> np.ndarray:
        """
        Perform rescale + normalize in-place, optionally in height-wise tiles,
        so we never allocate a second full-sized array like (H,W,C) float64.

        image is assumed to be float32 and in the *original* input_data_format
        (channels_first or channels_last).
        """
        if not (do_rescale or do_normalize):
            return image

        # Ensure mean/std are arrays for broadcasting
        mean_arr = np.asarray(mean, dtype=image.dtype)
        std_arr = np.asarray(std, dtype=image.dtype)

        # Channels-first: [C, H, W]
        if input_data_format in (ChannelDimension.FIRST, "channels_first"):
            C, H, W = image.shape

            if mean_arr.ndim == 1 and mean_arr.shape[0] == C:
                mean_bc = mean_arr.reshape(C, 1, 1)
                std_bc = std_arr.reshape(C, 1, 1)
            else:
                mean_bc = mean_arr
                std_bc = std_arr

            for y0 in range(0, H, tile_h):
                y1 = min(H, y0 + tile_h)
                tile = image[:, y0:y1, :]

                if do_rescale:
                    tile *= rescale_factor

                if do_normalize:
                    tile -= mean_bc
                    tile /= std_bc

        # Channels-last: [H, W, C]
        elif input_data_format in (ChannelDimension.LAST, "channels_last"):
            H, W, C = image.shape

            if mean_arr.ndim == 1 and mean_arr.shape[0] == C:
                mean_bc = mean_arr.reshape(1, 1, C)
                std_bc = std_arr.reshape(1, 1, C)
            else:
                mean_bc = mean_arr
                std_bc = std_arr

            for y0 in range(0, H, tile_h):
                y1 = min(H, y0 + tile_h)
                tile = image[y0:y1, :, :]

                if do_rescale:
                    tile *= rescale_factor

                if do_normalize:
                    tile -= mean_bc
                    tile /= std_bc

        else:
            # Fallback (should not normally happen)
            if do_rescale:
                image *= rescale_factor
            if do_normalize:
                image -= mean_arr
                image /= std_arr

        return image

    # ============================================================
    # NEW: per-frame patch embedding (no temporal 8D tensor)
    # ============================================================
    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        uigraph_use: bool = False,
        uigraph_diff: float = 0.0,
        uigraph_rand: bool = False,
    ):
        """
        Preprocess an image or a *sequence of frames*.

        IMPORTANT CHANGES:
        - We patch-embed *one frame at a time*, instead of building a big
          (T, ...) tensor and reshaping to 8D.
        - Each frame is resized (aggressively) so that HxW <= runtime_max_pixels.
        - Rescale + normalize are done in-place in float32; tokens are float16.
        - Temporal information is represented only as grid_t = num_frames in
          image_grid_thw; no temporal patch merging.
        """
        frames = make_list_of_images(images)

        if do_convert_rgb:
            frames = [convert_to_rgb(img) for img in frames]

        # Convert the *first* frame to numpy to infer format / base size
        first_arr = to_numpy_array(frames[0])

        if is_scaled_image(first_arr) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(first_arr)

        # Base size from first frame
        base_h, base_w = get_image_size(first_arr, channel_dim=input_data_format)

        # We will accumulate per-frame tokens here
        all_tokens: List[np.ndarray] = []
        processed_resize: List[tuple] = []

        # We'll compute grid_h/grid_w from the *first* processed frame and assume
        # all subsequent frames are resized the same way.
        grid_h = grid_w = None

        # Per-frame loop – this keeps peak RAM much lower
        for frame in frames:
            arr = to_numpy_array(frame)

            # 1. Aggressive resize: clamp by runtime_max_pixels (<=224x224)
            if do_resize:
                resized_h, resized_w = smart_resize(
                    base_h,
                    base_w,
                    factor=self.patch_size,   # spatial patch, not temporal
                    min_pixels=self.min_pixels,
                    max_pixels=self.runtime_max_pixels,  # 🔧 heavy clamp
                )
                arr = resize(
                    arr,
                    size=(resized_h, resized_w),
                    resample=resample,
                    input_data_format=input_data_format,
                )
            else:
                resized_h, resized_w = get_image_size(arr, channel_dim=input_data_format)

            # 2. float32 only
            arr = arr.astype(np.float32, copy=False)

            # 3. In-place rescale + normalize
            arr = self._rescale_and_normalize_inplace(
                arr,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                mean=image_mean,
                std=image_std,
                input_data_format=input_data_format,
                tile_h=256,
            )

            # 4. Convert to channels-first for patching: [C, H, W]
            arr_cf = to_channel_dimension_format(arr, ChannelDimension.FIRST, input_channel_dim=input_data_format)
            C, H, W = arr_cf.shape

            # Ensure dimensions are multiples of patch_size; we simply floor-crop
            ps = self.patch_size
            gh = H // ps
            gw = W // ps
            if gh == 0 or gw == 0:
                raise ValueError(f"Image too small after resizing: H={H}, W={W}, patch_size={ps}")

            H_used = gh * ps
            W_used = gw * ps
            if H_used != H or W_used != W:
                arr_cf = arr_cf[:, :H_used, :W_used]
                H, W = H_used, W_used

            # Save grid size from first frame
            if grid_h is None:
                grid_h, grid_w = gh, gw
            else:
                # Sanity: if later frames differ, we clamp to the min
                gh = min(grid_h, gh)
                gw = min(grid_w, gw)
                H_used = gh * ps
                W_used = gw * ps
                arr_cf = arr_cf[:, :H_used, :W_used]
                grid_h, grid_w = gh, gw

            # 5. Patchify this single frame:
            # arr_cf: [C, H_used, W_used]
            # -> [C, grid_h, ps, grid_w, ps]
            patches_frame = arr_cf.reshape(
                C,
                grid_h,
                ps,
                grid_w,
                ps,
            )  # [C, grid_h, ps, grid_w, ps]

            # -> [grid_h, grid_w, C, ps, ps]
            patches_frame = patches_frame.transpose(1, 3, 0, 2, 4)

            # -> tokens for this frame: [grid_h*grid_w, C*ps*ps]
            tokens_frame = patches_frame.reshape(
                grid_h * grid_w,
                C * ps * ps,
            ).astype(np.float16, copy=False)

            all_tokens.append(tokens_frame)
            processed_resize.append((H, W))

        # Concatenate all frame tokens: [T*grid_h*grid_w, C*ps*ps]
        if len(all_tokens) == 1:
            flatten_patches = all_tokens[0]
        else:
            flatten_patches = np.concatenate(all_tokens, axis=0)

        # Temporal length = number of frames
        grid_t = len(frames)

        # Default ui-graph: just identity indices per (t, h, w)
        grid_h_half = grid_h // self.merge_size
        grid_w_half = grid_w // self.merge_size

        # Note: we are NOT building the heavy uigraph here to save RAM.
        # If you really want it, you could reconstruct per-frame patches here,
        # but that would defeat the purpose of staying under memory limits.
        uigraph_assign = np.arange(grid_t * grid_h_half * grid_w_half).reshape(
            (grid_t, grid_h_half, grid_w_half)
        )

        return flatten_patches, (grid_t, grid_h, grid_w), uigraph_assign, processed_resize

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        uigraph_use: bool = False,
        uigraph_diff: float = 0.0,
        uigraph_rand: bool = False,
        vis_dir: str = None,
    ):
        """
        High-level preprocess entry.

        The main heavy lifting is now done by `_preprocess`, which:
        - works per frame,
        - uses strong downscaling and float16 tokens.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = make_batched_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        data = {}

        # --------- Image path (still images or frame lists treated as one sample) ----------
        if images is not None:
            pixel_values, vision_grid_thws = [], []

            patch_assign_sep = []        # per ui-graph, but we keep it simple now
            patch_assign_len = []
            patch_assign_shared = []

            for image in images:
                patches, image_grid_thw, uigraph_assign, image_resize = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    uigraph_use=uigraph_use,
                    uigraph_diff=uigraph_diff,
                    uigraph_rand=uigraph_rand,
                )

                if uigraph_use and uigraph_rand:
                    C = len(np.unique(uigraph_assign))
                    _, H, W = uigraph_assign.shape
                    uigraph_assign = np.random.randint(0, C + 1, size=(1, H, W))

                uigraph_assign_1d = uigraph_assign.flatten()
                uigraph_assign_1d = self.rerank_values(uigraph_assign_1d)
                uigraph_assign_len = len(np.unique(uigraph_assign_1d))

                uigraph_assign_1d += sum(patch_assign_len)
                patch_assign_shared.extend(uigraph_assign_1d)
                patch_assign_sep.extend(uigraph_assign_1d)
                patch_assign_len.append(uigraph_assign_len)

                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)

                if vis_dir is not None and uigraph_use:
                    image_vis = self._vis_uigraph(uigraph_assign, image_resize, self.patch_size * self.merge_size, image)
                    image_vis.save(f"{vis_dir}/demo.png")

            pixel_values = np.array(pixel_values, dtype=np.float16)
            vision_grid_thws = np.array(vision_grid_thws)
            patch_assign_shared = np.array(patch_assign_shared)

            data.update(
                {
                    "pixel_values": pixel_values,
                    "image_grid_thw": vision_grid_thws,
                    "patch_assign": patch_assign_shared,
                    "patch_assign_sep": patch_assign_sep,
                    "patch_assign_len": patch_assign_len,
                }
            )

        # --------- Video path (list of frames per video) ----------
        if videos is not None:
            pixel_values, vision_grid_thws = [], []
            for images_ in videos:
                # uigraph not support video yet in this low-RAM path; we ignore it for videos
                patches, video_grid_thw, _, _ = self._preprocess(
                    images_,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    uigraph_use=False,  # force off for safety
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)

            pixel_values = np.array(pixel_values, dtype=np.float16)
            vision_grid_thws = np.array(vision_grid_thws)
            data.update(
                {
                    "pixel_values_videos": pixel_values,
                    "video_grid_thw": vision_grid_thws,
                }
            )

        return BatchFeature(data=data, tensor_type=return_tensors)
