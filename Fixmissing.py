import os
import pyreadr
import pandas as pd

rda = r'C:\Users\Sobia Khan\Downloads\DATA-adni\ADNIMERGE2\ADNIMERGE2\data'
out = r'C:\Users\Sobia Khan\Downloads\DATA-adni\ADNIMERGE_CSVs'

os.makedirs(out, exist_ok=True)

# Fix 1: FreeSurfer 7.x
print("Looking for FreeSurfer 7.x...")
for name in ['UCSFFSX7', 'UCSFSX7', 'UCSFFSX6']:
    p = os.path.join(rda, name + '.rda')
    if os.path.exists(p):
        result = pyreadr.read_r(p)
        for k, df in result.items():
            df.to_csv(os.path.join(out, k + '.csv'), index=False)
            print(f'  ✅ {k}.csv  {df.shape}')
        break
    else:
        print(f'  trying {name}...')

# Fix 2: Amyloid PET
print("\nLooking for Amyloid PET...")
for name in ['UCBERKELEYAMV_6MM', 'UCBERKELEYAV45_6MM',
             'UCBERKELEYAMY_6MM', 'UCBERKELEY_AMY_6MM',
             'UCBERKELEYAV_6MM']:
    p = os.path.join(rda, name + '.rda')
    if os.path.exists(p):
        result = pyreadr.read_r(p)
        for k, df in result.items():
            df.to_csv(os.path.join(out, k + '.csv'), index=False)
            print(f'  ✅ {k}.csv  {df.shape}')
        break
    else:
        print(f'  trying {name}...')

# Show all files in rda folder containing "AMY" or "AMV" or "AV45"
print("\nSearching for any Amyloid-related .rda files in your data folder...")
for f in sorted(os.listdir(rda)):
    if any(x in f.upper() for x in ['AMY', 'AMV', 'AV45', 'AMYLOID']):
        print(f'  Found: {f}')

print("\nSearching for any FreeSurfer 7 .rda files...")
for f in sorted(os.listdir(rda)):
    if 'UCSFFS' in f.upper() or 'UCSFSX' in f.upper():
        print(f'  Found: {f}')

print('\nDone!')