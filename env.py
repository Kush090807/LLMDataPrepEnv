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
        self.conversion_rates = {
            "EUR": 1.10,
            "£": 1.25,
            "$": 1.0,
            "USD": 1.0,
            "A$": 0.65
        }
        self.current_data = []
        self.step_count = 0
        self.max_steps = 50

    def _load_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "data.json")
        with open(data_path, "r") as f:
            return json.load(f)

    def reset(self) -> Observation:
        self.current_data = copy.deepcopy(self.original_data)
        self.step_count = 0
        return self.state()

    def state(self) -> Observation:
        return Observation(
            dataset=self.current_data,
            target_schema=self.schema,
            conversion_rates=self.conversion_rates
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        reward_val = 0.0
        done = False
        info = {"error": None}

        if self.step_count >= self.max_steps:
            done = True

        if isinstance(action, DeleteRowAction):
            if 0 <= action.row_index < len(self.current_data):
                row = self.current_data[action.row_index]
                if row.get("email") is None:
                    # Good action: deleted null email row
                    reward_val += 0.2
                else:
                    # Bad action: deleted a valid row
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
                    # syntax reward for valid target
                    reward_val += 0.1
                else:
                    reward_val -= 0.1
                    info["error"] = f"Invalid column: {action.column}"
            else:
                reward_val -= 0.1
                info["error"] = "Invalid row_index"
                
        elif isinstance(action, PassAction):
            done = True

        return self.state(), Reward(value=reward_val), done, info
