from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Literal # <--- ADDED Union and Literal

class Observation(BaseModel):
    dataset: List[Dict[str, Any]]
    target_schema: Dict[str, str]
    conversion_rates: Dict[str, float]
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

# This defines the set of allowed actions
Action = Union[DeleteRowAction, UpdateValueAction, PassAction]

class Reward(BaseModel):
    value: float
