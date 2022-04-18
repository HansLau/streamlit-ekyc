import glob, pip
import subprocess

for path in glob.glob("C:/Users/XL204978/Desktop/all_windows_file/all_windows_file/installation files/New folder/*.whl"):
    subprocess.run(f'pip install {path}')