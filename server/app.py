import os
from typing import Union
from openenv.core.env_server import create_fastapi_app
from env import LLMDataPrepEnv
from models import UpdateValueAction, DeleteRowAction, PassAction, Observation

# 1. Define your actions (Must match your models.py)
ActionType = Union[UpdateValueAction, DeleteRowAction, PassAction]

# 2. Initialize your environment class
env = LLMDataPrepEnv()

# 3. Create the FastAPI app using the OpenEnv helper
# This is the "magic" that makes your code a web server
app = create_fastapi_app(env, ActionType, Observation)

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces MUST run on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)