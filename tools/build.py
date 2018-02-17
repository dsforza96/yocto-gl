#! /usr/bin/env python3 -B

# build utility for easy development
# complete and unreliable hack used for making it easier to develop

import click, os, platform, markdown, glob, textwrap

def build(target, dirname, buildtype, cmakeopts=''):
    os.system('mkdir -p build/{dirname}; cd build/{dirname}; cmake ../../ -GNinja -DCMAKE_BUILD_TYPE={buildtype} -DYOCTO_EXPERIMENTAL=ON {cmakeopts}; cmake --build . {target}'.format(target=target, dirname=dirname, buildtype=buildtype, cmakeopts=cmakeopts))
    os.system('ln -Ffs {dirname} build/latest'.format(dirname=dirname))

@click.group()
def run():
    pass

@run.command()
@click.argument('target', required=False, default='')
def latest(target=''):
    os.system('cd build/latest; cmake --build . {target}'.format(target=target))

@run.command()
@click.argument('target', required=False, default='')
def release(target=''):
    build(target, 'release', 'Release')

@run.command()
@click.argument('target', required=False, default='')
def nogl(target=''):
    build(target, 'nogl', 'Release', '-DYOCTO_OPENGL=OFF')

@run.command()
@click.argument('target', required=False, default='')
def debug(target=''):
    build(target, 'debug', 'Debug')

@run.command()
@click.argument('target', required=False, default='')
def gcc(target=''):
    build(target, 'gcc', 'Release', '-DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7')

@run.command()
def xcode():
    os.system('mkdir -p build/xcode; cd build/xcode; cmake -G Xcode -DYOCTO_EXPERIMENTAL=ON ../../; open yocto-gl.xcodeproj')

@run.command()
def clean():
    os.system('rm -rf bin; rm -rf build')

@run.command()
def format():
    for glob in ['yocto/yocto_*.h', 'yocto/yocto_*.cpp', 'apps/y*.cpp']:
        os.system('clang-format -i -style=file ' + glob)

@run.command()
def docs():
    os.system('./tools/cpp2doc.py')

@run.command()
def doxygen():
    os.system('doxygen ./tools/Doxyfile')

@run.command()
@click.argument('msg',required=True)
def commit(msg):
    print('updating version ...')
    ncpp = ''
    with open('yocto/yocto_gl.h') as f:
        for line in f:
            if line.startswith('/// The current version'):
                tokens = line.split('.')
                tokens[-2] = str(int(tokens[-2])+1)
                line = '.'.join(tokens)
            ncpp += line
    with open('yocto/yocto_gl.h', 'wt') as f: f.write(ncpp)
    print('formatting code ...')
    os.system('./tools/build.py format')
    print('building docs ...')
    os.system('./tools/build.py docs')
    print('committing code ...')
    os.system(f'git commit -a -m "{msg}"')

if __name__ == '__main__':
    run()
