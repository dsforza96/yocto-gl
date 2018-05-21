# Notes on future improvements of Yocto/GL

This file contains notes on future improvements of Yocto.
Please consider this to be just development notes and not any real planning.

## Main features

- better tests
- better material rendering
- better rendering

## planned cleanup

- remove texture info back into texture
    - consider only wrap
- more image operations to floats
    - make explicit the use of 1/3/4 channel images
    - bumps and dimples should be sphere
- single material representation converted on load
- convert textures on load
    - srgb -> float on load with correct convention

## image to float

- should remove entirely metallic-roughness on input or provide two materials
- add textures
    - opacity -> needs splitting
    - occlusion -> needs splitting
    - metallic -> ks
    - roughness -> needs splitting
- obj -> scene
    - load textures as you go
    - split textures on the fly
- gltf -> scene
    - load textures as you go
    - split textures on the fly
    - convert specular-glossness to metallic roughness
        - probably recombine textures
- scene -> obj
    - no texture splitting is necessary
        - opacity remains opacity
        - use extension for roughness
- scene -> glTF
    - repack textures together
        - opacity -> diffuse
        - roughness -> specular
- on load, split appropriate textures with given ones
- handle images with vec3f/vec4f and combined vec3f+float
- example images have very few defaults
- move example images to vec3f/vec4f
- update yimproc
- move examples to vec3f/vec4f
- move saving to explicit srgb
- move loading to explicit srgb
- move image ops to float
- move image viewer to float
- move image operations to float
- move textures to float

## split material

- remap material values
- split textures on load

## Trace

- camera
    - thin lens camera - read pbrt
    - realistic camera - read pbrt
    - leave ray or camera_pos, lens_pos?
- clarify light pdfs
- one path
    - compute light pdfs with intersection
- orthonormal basis
    - file:///Users/fabio/Downloads/Duff2017Basis.pdf
- volumes
    - make material models clear
    - start with a simple design
- redirect log to imgui
- pass gltf model
- bump mapping
    - convert bump maps to normal maps internally on load
- displacement
    - apply displacement on tesselation or on triangle vertices directly
- examples
    - hair
    - subdiv
    - displacement
- put back double sided
    - one sided shading
- refraction
    - rough surface refraction
- intersection
    - methods with projections?
        - why shear?
        - triangle
        - line
        - point
    - line/point parametrization that mnatches pbrt and embree
    - radius in offsetting rays
        - check embree to avoid 
- bvh
    - sah
    - opacity
    - embree integration
- see mini path tracer features and include them all
- camera
    - realistic camera
    - thin lens camera
- simple denoiser
    - joint bilateral denoiser
    - non-local means denoiser
    - denoiser from paris

## Simpler shapes

- sampling shapes
    - remove sample_points, sample_lines, etc...
    - support uniform sampling in sample_shape

## glTF tests

- BUG: transforms
- not supported
    - alpha cutoff
    - texture modes
- yitrace crash
    - morph
    - sparse accessor
- update normals/tangents
- yview
    - 2CylinderEngine: near plane
    - AlphaBlendModeTest: not supported
    - AnimatedMorphCube/AnimatedMorphSphere: not supported
    - AnimatedTriangle: crash
    - Avocado: bad specular?
    - BarracubeFish: flip normal map? specular?
        - occlusion not supported
    - BoomBox: specular?
    - BoxVertexColor: check alpha?
    - Buggy: transforms
    - DamagedHelmet: no colors
    - Duck: not visible 
    - MetalRoughSphere: check with envmap
    - NormalTangentMirrorTest: bad normal and bad sideness
    - NormalTangentTestL check colors
    - SimpleMeshes: crash
    - SimpleMorph: crash
    - SimpleSparseAccessor: crash
    - Suzanne: bad materials
    - Triangle: crash
    - TextureSettingTest: bad texture and double-sided
    - VC: bad transforms

## Port scenes

- bitterli
    - hair
    - render exclude list
    - kitchen
        - bump map
    - car
        - shading bug
        - enable kr
        - metallic paint
    - car2
        - shading bug
        - enable kr
        - metallic paint
    - flip double sided normals on at least large objects
- mcguire
    - exclude list
    - bad textures
    - bad emission
        - lost empire
        - sportscar
    - cameras
    - lights
- pbrt
    - crown material or re-export
    - landscape materials
    - pbrt include parser
- gltf exports

## Test scenes: simnplify shape generation

- remove some tesselation function
- use subdivs for shader ball
- substance-like shader ball
- bent floor
- 0 roughness

## OpenGL/Trace

- investigate bump map on GPU
    - https://www.opengl.org/discussion_boards/showthread.php/162857-Computing-the-tangent-space-in-the-fragment-shader
    - http://jbit.net/~sparky/sfgrad_bump/mm_sfgrad_bump.pdf

## yView

- double sided in OBJ
- cutout not working on yview
    - consider cutout by default

## BVH

- SAH based build?
- simplify build code: can we avoid preallocating nodes?
- move away from special functions in BVH?
    - always use sort
    - provide a sort buffer
- add cutout to trace
- simplify build functions: 
- maybe put axis with internal
- simplify partition and nth_element function
    - include wrapper functions

## Math

- consider removing const refs
- frame inverse with flag
- check random shuffle
- check random number generation for float/double
