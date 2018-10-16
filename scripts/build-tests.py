#! /usr/bin/env python3 -B

import os, glob

for ext in ['obj', 'gltf', 'json', 'ybin', 'pbrt']:
    os.system(f'rm -rf tests/{ext}; mkdir tests/{ext}')
for filename in glob.glob('tests/*.json'):
    print(filename)
    basename = os.path.basename(filename).replace('.json','')
    for ext in ['obj', 'gltf', 'json', 'ybin', 'pbrt']:
        os.system(f'mkdir tests/{ext}/{basename}')
        os.system(f'mkdir tests/{ext}/{basename}/textures')
        os.system(f'mkdir tests/{ext}/{basename}/meshes')
        os.system(f'mkdir tests/{ext}/{basename}/models')
        os.system(f'./bin/yscnproc -o tests/{ext}/{basename}/{basename}.{ext} {filename}')
