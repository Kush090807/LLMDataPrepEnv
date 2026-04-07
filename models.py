from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Observation(BaseModel):
    dataset: List[Dict[str, Any]]
    target_schema: Dict[str, str]
    conversion_rates: Dict[str, float]
    # ADD THIS LINE BELOW
    reward: Optional[float] = 0.0

class DeleteRowAction(BaseModel):
    action_type: Literal["DeleteRowAction"] = "DeleteRowAction"
    row_index: int
    reason: str

class UpdateValueAction(BaseModel):
    action_type: Literal["UpdateValueAction"] = "UpdateValueAction"
    row_index: int
    column: str
    new_value: Any

class PassAction(BaseModel):
    action_type: Literal["PassAction"] = "PassAction"

# Optional: Add an explicit type field or use discriminator if desired, but this works fine for openai function calling
Action = Union[DeleteRowAction, UpdateValueAction, PassAction]

class Reward(BaseModel):
    value: float
