from pydantic import BaseModel, field_validator
from typing import Optional


class EmbedRequest(BaseModel):
    # Prompt
    inputs: str
    truncate: bool = False

    @field_validator("inputs")
    def valid_input(cls, v):
        if not v:
            raise ValueError("`inputs` cannot be empty")
        return v


# `info` return value
class InfoResponse(BaseModel):
    # Model info
    model_id: str
    model_sha: Optional[str]
    model_dtype: str
    model_pooling: Optional[str]

    # Router Parameters
    max_concurrent_requests: int
    max_input_length: int
    max_batch_tokens: int
    max_batch_requests: Optional[int]
    max_client_batch_size: int
    tokenization_workers: int

    # Router Info
    version: str
    sha: Optional[str]
    docker_label: Optional[str]


# Inference API currently deployed model
class DeployedModel(BaseModel):
    model_id: str
    sha: str
