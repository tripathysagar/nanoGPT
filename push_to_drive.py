import subprocess

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
