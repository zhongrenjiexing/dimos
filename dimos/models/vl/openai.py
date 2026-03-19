from dataclasses import dataclass
from functools import cached_property
import os
from typing import Any

import numpy as np
from openai import OpenAI

from dimos.models.vl.base import VlModel, VlModelConfig
from dimos.msgs.sensor_msgs import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class OpenAIVlModelConfig(VlModelConfig):
    model_name: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None  # e.g. http://server:8011/v1 for OpenAI-compatible servers


class OpenAIVlModel(VlModel):
    default_config = OpenAIVlModelConfig
    config: OpenAIVlModelConfig

    @cached_property
    def _client(self) -> OpenAI:
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.config.base_url or os.getenv("OPENAI_BASE_URL")
        if not api_key and not base_url:
            raise ValueError(
                "OPENAI_API_KEY or OPENAI_BASE_URL must be set, and no vlm instance provided"
            )
        kwargs: dict[str, str] = {"api_key": api_key or "not-needed"}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        return OpenAI(**kwargs)

    def query(self, image: Image | np.ndarray, query: str, response_format: dict | None = None, **kwargs) -> str:  # type: ignore[override, type-arg, no-untyped-def]
        if isinstance(image, np.ndarray):
            import warnings

            warnings.warn(
                "OpenAIVlModel.query should receive standard dimos Image type, not a numpy array",
                DeprecationWarning,
                stacklevel=2,
            )

            image = Image.from_numpy(image)

        # Apply auto_resize if configured
        image, _ = self._prepare_image(image)

        img_base64 = image.to_base64()

        api_kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ],
        }

        if response_format:
            api_kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**api_kwargs)

        return response.choices[0].message.content  # type: ignore[return-value,no-any-return]

    def query_batch(
        self, images: list[Image], query: str, response_format: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[str]:  # type: ignore[override]
        """Query VLM with multiple images using a single API call."""
        if not images:
            return []

        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._prepare_image(img)[0].to_base64()}"},
            }
            for img in images
        ]
        content.append({"type": "text", "text": query})

        messages = [{"role": "user", "content": content}]
        api_kwargs: dict[str, Any] = {"model": self.config.model_name, "messages": messages}
        if response_format:
            api_kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**api_kwargs)
        response_text = response.choices[0].message.content or ""
        # Return one response per image (same response since API analyzes all images together)
        return [response_text] * len(images)

    def stop(self) -> None:
        """Release the OpenAI client."""
        if "_client" in self.__dict__:
            del self.__dict__["_client"]

