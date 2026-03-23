import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import albumentations as A
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAugmenter:
    def __init__(
        self,
        augmentations_per_image: int = 2,
        seed: int = 42,
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions

        self._set_seed()

        self.transform = A.Compose(
            [
                A.Rotate(limit=15, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=0,
                    p=0.8,
                    border_mode=0,
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MotionBlur(blur_limit=7, p=0.5),
                    ],
                    p=0.3,
                ),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )
    
    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def augment_image(self, image: Image.Image) -> Image.Image:
        image_np = np.array(image)

        augmented = self.transform(image=image_np)
        augmented_image_np = augmented["image"]

        return Image.fromarray(augmented_image_np.astype(np.uint8))

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        count = 0

        image_files = self._find_image_files(input_path)

        logger.info(f"Found {len(image_files)} images to augment.")

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

            rel_dir = img_path.parent.relative_to(input_path)
            target_dir = output_path / rel_dir
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)

            if self.save_original:
                orig_name = f"orig_{img_path.name}"
                image.save(target_dir / orig_name)

            for i in range(self.augmentations_per_image):
                augmented = self.augment_image(image.copy())
                aug_name = f"aug_{i}_{img_path.name}"
                augmented.save(target_dir / aug_name)
                count += 1

        logger.info(
            f"Augmentation of {count} images completed. Output saved to: {output_dir}"
        )

    def _find_image_files(self, root: Path) -> List[Path]:
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 2,
    seed: int = 42,
) -> None:
    augmenter = ImageAugmenter(
        augmentations_per_image=augmentations_per_image, seed=seed, save_original=True
    )
    augmenter.process_directory(input_dir, output_dir)
