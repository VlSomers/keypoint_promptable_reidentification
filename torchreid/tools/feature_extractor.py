from __future__ import absolute_import
import torch
import torchvision.transforms as T
import copy

from torchreid.data import ImageDataset
from torchreid.data.datasets.keypoints_to_masks import KeypointsToMasks
from torchreid.scripts.builder import build_model
from torchreid.utils import compute_model_complexity
from torchreid.data.transforms import build_transforms
from torchreid.utils.tools import extract_test_embeddings


class KPRFeatureExtractor(object):
    """A simple API for feature extraction derived from the original Torchreid one.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

        from torchreid.utils import FeatureExtractor

        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        image_list = [
            'a/b/c/image001.jpg',
            'a/b/c/image002.jpg',
            'a/b/c/image003.jpg',
            'a/b/c/image004.jpg',
            'a/b/c/image005.jpg'
        ]

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        cfg,
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        verbose=True,
        model=None,
    ):

        # Build model
        if model is None:
            model = build_model(cfg, verbose=verbose)

        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(model, cfg)
            print("Model: {}".format(cfg.model.name))
            print("- params: {:,}".format(num_params))
            print("- flops: {:,}".format(flops))

        # Build transform functions
        _, preprocess, self.target_preprocess, self.prompt_preprocess = build_transforms(
            image_size[0],
            image_size[1],
            cfg,
            transforms=None,
            norm_mean=pixel_mean,
            norm_std=pixel_std,
            masks_preprocess=cfg.model.kpr.masks.preprocess,
            softmax_weight=cfg.model.kpr.masks.softmax_weight,
            background_computation_strategy=cfg.model.kpr.masks.background_computation_strategy,
            mask_filtering_threshold=cfg.model.kpr.masks.mask_filtering_threshold,
            verbose=verbose,
        )

        to_pil = T.ToPILImage()

        self.keypoints_to_prompt_masks = KeypointsToMasks(mode=cfg.model.kpr.keypoints.prompt_masks,
                                                          vis_thresh=cfg.model.kpr.keypoints.vis_thresh,
                                                          vis_continous=cfg.model.kpr.keypoints.vis_continous,
                                                          )
        self.keypoints_to_target_masks = KeypointsToMasks(mode=cfg.model.kpr.keypoints.target_masks,
                                                          vis_thresh=cfg.model.kpr.keypoints.vis_thresh,
                                                          vis_continous=False,
                                                          )


        # Class attributes
        self.cfg = cfg
        self.model = model
        self.device = "cuda" if cfg.use_gpu else "cpu"
        self.preprocess = preprocess
        self.to_pil = to_pil

    def __call__(self, input):
        # Convert input to a list if it's a single sample (dict)
        samples = [input] if isinstance(input, dict) else input

        # Deep copy of samples
        updated_samples = copy.deepcopy(samples)

        # Initialize batch dictionary with empty lists
        batch = {"image": [], "prompt_masks": []}

        # Iterate over the list of samples
        for sample in samples:
            preprocessed_sample = ImageDataset.getitem(
                sample,
                self.cfg,
                self.keypoints_to_prompt_masks,
                self.prompt_preprocess,
                self.keypoints_to_target_masks,
                self.target_preprocess,
                self.preprocess,
                load_masks=True,
            )

            # Append the preprocessed image and prompt_masks to the batch
            batch["image"].append(preprocessed_sample["image"])
            if "prompt_masks" in preprocessed_sample:
                batch["prompt_masks"].append(preprocessed_sample["prompt_masks"])

        # Convert lists to tensors and concatenate along the batch dimension
        batch["image"] = torch.stack(batch["image"], dim=0)
        if len(batch["prompt_masks"]) > 0:
            batch["prompt_masks"] = torch.stack(batch["prompt_masks"], dim=0)

        args = {}
        args["images"] = batch["image"]
        if "prompt_masks" in batch:
            args["prompt_masks"] = batch["prompt_masks"]

        # Forward pass through the model
        model_output = self.model(**args)

        # Extract embeddings and other outputs
        (
            embeddings,
            visibility_scores,
            parts_masks,
            pixels_cls_scores,
        ) = extract_test_embeddings(
            model_output, self.cfg.model.kpr.test_embeddings
        )

        if self.cfg.test.normalize_feature:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Go through embeddings, visibility_scores and parts_masks and put them back into the samples
        for i in range(len(updated_samples)):
            updated_samples[i]["embeddings"] = embeddings[i].detach().cpu().numpy()
            updated_samples[i]["visibility_scores"] = visibility_scores[i].detach().cpu().numpy()
            updated_samples[i]["parts_masks"] = parts_masks[i].detach().cpu().numpy()

        return updated_samples, embeddings, visibility_scores, parts_masks
