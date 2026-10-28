"""Model configuration shared by every agent in this lab."""
import os

# The finance edition used gpt-4o for every agent. Override with CC_OPENAI_MODEL.
MODEL = os.getenv("CC_OPENAI_MODEL", "gpt-4o")
