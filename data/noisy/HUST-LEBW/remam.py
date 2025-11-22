import os, re

for name in os.listdir('.'):
    m = re.match(r'^video(\d+)(\.[^.]+)$', name, flags=re.IGNORECASE)
    if m:
        new = f"v{int(m.group(1))}{m.group(2)}"  # int() removes any leading zeros
        if new != name:
            print(f"{name} -> {new}")
            os.rename(name, new)
