import subprocess
for instanceNumber in range(3, 12):
    subprocess.call(['python3', 'bettiScript.py', str(instanceNumber)])
