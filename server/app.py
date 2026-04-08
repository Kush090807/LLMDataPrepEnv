import os
from typing import Union
from openenv.core.env_server import create_fastapi_app
from env import LLMDataPrepEnv
from models import UpdateValueAction, DeleteRowAction, PassAction, Observation


ActionType = Union[UpdateValueAction, DeleteRowAction, PassAction]


app = create_fastapi_app(LLMDataPrepEnv, ActionType, Observation)

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces MUST run on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
