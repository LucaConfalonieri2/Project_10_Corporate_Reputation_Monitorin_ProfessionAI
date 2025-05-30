from huggingface_hub import create_repo, upload_folder
import utils

# Crea il repo
create_repo(utils.REPO_ID, exist_ok=True, private=True)

# Pusha il repo
upload_folder(
    folder_path=utils.MODEL_PATH,
    repo_id=utils.REPO_ID,
    repo_type="model",
    commit_message="upload automatico del modello"
)

