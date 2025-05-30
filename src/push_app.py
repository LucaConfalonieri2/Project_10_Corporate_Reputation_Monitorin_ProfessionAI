from huggingface_hub import HfApi
import os
REPO_ID = "confa3452/fasttext-sentiment-it-ProfectionAI"

api = HfApi()

api.upload_folder(
    repo_id=REPO_ID,
    folder_path="spaces",
    repo_type="model",
    token=os.getenv("HF_TOKEN"),
    commit_message="Deploy automatico della demo app"
)
