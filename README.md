# Ray Tracing in One Weekend

![Output Image](assets/image.png)

Didactic ray tracing implementation using C++ adapted from [Ray Tracing in One Weekend][1] by [Peter Shirley][2]. Apart from the original implementation, I've tried to achieve parallelism using Cuda C++.

## Requirements

- g++ 7.4.0 or Cuda Toolkit with nvcc v9.1
- make 4.1

## Compilation and Running

```md
1. Clone the repository: $ git clone https://github.com/r1walz/ray-tracing-in-one-weekend.git
2. cd ray-tracing-in-one-weekend
3. use `$ make` to compile using g++ or `$ make CC=nvcc` for gpu parallelism
4. ./tracer >image.ppm
5. Open the `image.ppm` file using your favourite ppm image viewer
```

To clean up, use `$ make clean`.

[1]: https://github.com/RayTracing/raytracing.github.io
[2]: https://research.nvidia.com/person/peter-shirley
