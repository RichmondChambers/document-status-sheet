import os
import openai
from knowledge_base import rebuild_knowledge_base

openai.api_key = os.environ.get("OPENAI_API_KEY")

if __name__ == "__main__":
    rebuild_knowledge_base()
    print("Knowledge base rebuild complete.")
