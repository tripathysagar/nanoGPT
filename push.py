import subprocess
import sys
import shutil
import os
from huggingface_hub import Repository
from fastcore.xtras import Path
import fire

from model import GPT, GPTConfig

ckpt_path = Path('out/ckpt.pt')
model_fn = 'out/model.pt'
token_fn = 'data_src/tokenizer_bpe.json'

def push_to_hub(HF_TOKEN):
    if not ckpt_path.exists() or HF_TOKEN == None:
        sys.exit()

    GPT.save_model_from_file(ckpt_path=ckpt_path, out_dir='out/')


    repo_id = "tripathysagar/oida-gpt"  
    local_repo_path = "./tmp"  

    repo = Repository(
        local_dir=local_repo_path,
        clone_from=f"https://huggingface.co/{repo_id}",
        token=HF_TOKEN
    )

    shutil.move(model_fn, f"{local_repo_path}/model.pt")
    shutil.move(token_fn, f"{local_repo_path}/tokenizer.json")

    repo.push_to_hub(commit_message=f"Add tensor file: {token_fn} and {model_fn}")



"""
try:
    from google.colab import drive
    drive.mount('/content/gdrive')
except ImportError:
    print("Google Colab is not available. Drive mount skipped.")

def to_drive(local_file="/content/nanoGPT/out/ckpt.pt", 
             destination_path="/content/gdrive/My Drive/"):
    # Check if the file already exists to avoid duplicates
    try:
        # Using -f flag to overwrite if file exists
        subprocess.run(["cp", "-f", local_file, destination_path], check=True)
        print(f"Copy completed of {local_file}. Existing file will be overwritten if present.")
    except subprocess.CalledProcessError as e:
        print(f"Error while copying file: {e}")
    
# Call the function directly in the script if needed
# to_drive()

# Ensure the function is ready to be imported and used
if __name__ == "__main__":
    print("Running the script directly!")
    to_drive()
"""

if __name__ == '__main__':
  fire.Fire(push_to_hub)