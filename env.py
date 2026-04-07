import json
import os
import copy
from typing import Tuple, Dict, Any
from models import Observation, Action, Reward, DeleteRowAction, UpdateValueAction, PassAction

class LLMDataPrepEnv:
    def __init__(self):
        self.original_data = self._load_data()
        self.schema = {
            "id": "int",
            "name": "str",
            "email": "str (must not be null)",
            "signup_date": "str (ISO format YYYY-MM-DD)",
            "purchase_total": "float (in USD)"
        }
        self.conversion_rates = {"EUR": 1.10, "£": 1.25, "$": 1.0, "USD": 1.0, "A$": 0.65}
        self.current_data = []
        self.step_count = 0
        self.max_steps = 50

    def _load_data(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "data.json")
        with open(data_path, "r") as f:
            return json.load(f)

    # UPDATED: Passes 0.0 reward on reset
    async def reset(self, **kwargs) -> Observation:
        self.current_data = copy.deepcopy(self.original_data)
        self.step_count = 0
        return await self.state(reward=0.0)

    # UPDATED: Accepts reward and puts it in the Observation object
    async def state(self, reward: float = 0.0) -> Observation:
        return Observation(
            dataset=self.current_data,
            target_schema=self.schema,
            conversion_rates=self.conversion_rates,
            reward=reward  # <--- This connects to your models.py change
        )

    async def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        reward_val, done, info = 0.0, False, {"error": None}
        if self.step_count >= self.max_steps:
            done = True

        # --- Your existing action logic ---
        if isinstance(action, DeleteRowAction):
            if 0 <= action.row_index < len(self.current_data):
                row = self.current_data[action.row_index]
                if row.get("email") is None:
                    reward_val += 0.2
                else:
                    reward_val -= 0.1
                self.current_data.pop(action.row_index)
            else:
                reward_val -= 0.1
                info["error"] = "Invalid row_index"
        
        elif isinstance(action, UpdateValueAction):
            if 0 <= action.row_index < len(self.current_data):
                row = self.current_data[action.row_index]
                if action.column in row:
                    row[action.column] = action.new_value
                    reward_val += 0.1
                else: reward_val -= 0.1
            else:
                reward_val -= 0.1
                info["error"] = "Invalid row_index"
                
        elif isinstance(action, PassAction):
            done = True

        # UPDATED: Returns state with the calculated reward_val
        return await self.state(reward=reward_val), Reward(value=reward_val), done, info

    async def reset_async(self, **kwargs):
        return await self.reset(**kwargs)

    async def step_async(self, action):
        return await self.step(action)

    def close(self):
        pass
