import subprocess
for instanceNumber in range(12):
    subprocess.call(['python3', 'oldBettiScript.py', str(instanceNumber)])
