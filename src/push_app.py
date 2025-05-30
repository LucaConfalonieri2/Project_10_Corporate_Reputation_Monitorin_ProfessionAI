from huggingface_hub import HfApi
import os
import utils

api = HfApi()

api.upload_folder(
    repo_id=utils.REPO_ID,
    folder_path="spaces",
    repo_type="space",
    token=os.getenv("HF_TOKEN"),
    commit_message="Deploy automatico della demo app"
)
