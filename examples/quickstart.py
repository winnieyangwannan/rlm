import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

# Create the RLM Instance
rlm = RLM(
    backend="azure_openai",
    backend_kwargs={
        "model_name": "gpt-5",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "api_version": "2025-03-01-preview",
    },
    environment="local",  # Using local since Docker isn't available
    environment_kwargs={},
    max_depth=1,
    logger=logger,
    verbose=True,  # For printing to console with rich, disabled by default.
)

result = rlm.completion("Print me the first 5 powers of two, each on a newline.")

print(result)
