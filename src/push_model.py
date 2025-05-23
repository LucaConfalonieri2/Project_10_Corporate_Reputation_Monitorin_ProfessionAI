from huggingface_hub import create_repo, upload_folder
import utils

# 1. Crea il repo
create_repo(utils.REPO_ID, exist_ok=True, private=True)

# 2. Pusha il repo
upload_folder(
    folder_path=utils.MODEL_PATH,
    repo_id=utils.REPO_ID,
    repo_type="model",
    commit_message="upload automatico del modello"
)

print("Modello caricato su huggingface.")
