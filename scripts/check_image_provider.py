"""Verify Hugging Face image inference and save one test image."""

import asyncio
import base64
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from seo_blog_assistant.config import Settings
from seo_blog_assistant.providers.image import (
    ImageProviderError,
    create_image_provider,
)


def main() -> int:
    output_path = Path.cwd() / "test_image.png"
    if output_path.exists():
        print("ERROR test_image.png already exists; choose an empty directory.")
        return 1

    try:
        settings = Settings()
        provider = create_image_provider(settings)
        generated = asyncio.run(
            provider.generate_image(
                "a red apple on a wooden table, studio photo",
                "A red apple on a wooden table",
            )
        )
        image_bytes = base64.b64decode(generated.image_base64, validate=True)
        with Image.open(BytesIO(image_bytes)) as image:
            dimensions = image.size
        with output_path.open("xb") as image_file:
            image_file.write(image_bytes)
    except ImageProviderError as exc:
        print(f"ERROR {exc}")
        return 1
    except Exception as exc:
        print(f"ERROR image provider self-check failed ({type(exc).__name__}).")
        return 1

    print(f"OK {dimensions[0]}x{dimensions[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
