#! /usr/bin/env python3 -B

import click, glob, os

@click.group()
def cli():
    pass

@cli.command()
def release():
    os.system('mkdir -p build && mkdir -p build/release && cd build/release && cmake ../.. -GNinja -DYOCTO_EMBREE=ON && ninja')

@cli.command()
def debug():
    os.system('mkdir -p build && mkdir -p build/debug && cd build/debug && cmake ../.. -GNinja -DYOCTO_EMBREE=ON -DCMAKE_BUILD_TYPE=Debug && ninja')

@cli.command()
def xcode():
    os.system('mkdir -p build && mkdir -p build/xcode && cd build/xcode && cmake ../.. -GXcode -DYOCTO_EMBREE=ON && open yocto-gl.xcodeproj')

@cli.command()
def clean():
    os.system('rm -rf bin && rm -rf build')

@cli.command()
def test():
    os.system('mkdir -p build && mkdir -p build/release && cd build/release && cmake ../.. -GNinja -DYOCTO_EMBREE=ON')
    os.system('rm tests/run-tests/output/*.png && rm tests/run-tests/difference/*.png && ctest -j 4 --output-on-failure')    

@cli.command()
def tests():
    for ext in ['obj', 'gltf', 'json', 'ybin', 'pbrt']:
        os.system(f'rm -rf tests/{ext}; mkdir tests/{ext}')
    for filename in glob.glob('tests/*.json'):
        print(filename)
        basename = os.path.basename(filename).replace('.json','')
        for ext in ['obj', 'gltf', 'json', 'ybin', 'pbrt']:
            os.system(f'mkdir -p tests/{ext}/{basename}')
            os.system(f'mkdir -p tests/{ext}/{basename}/textures')
            os.system(f'mkdir -p tests/{ext}/{basename}/meshes')
            os.system(f'mkdir -p tests/{ext}/{basename}/models')
            os.system(f'./bin/yscnproc -o tests/{ext}/{basename}/{basename}.{ext} {filename}')

@cli.command()
def format():
    os.system('clang-format -i -style=file yocto/y*.h')
    os.system('clang-format -i -style=file yocto/y*.cpp')
    os.system('clang-format -i -style=file apps/y*.h')
    os.system('clang-format -i -style=file apps/y*.cpp')

cli()