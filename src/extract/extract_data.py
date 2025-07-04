from google.colab import drive

def mount_drive():
    """
    Mount Google Drive to access files stored there.
    """
    drive.mount('/content/drive')
    