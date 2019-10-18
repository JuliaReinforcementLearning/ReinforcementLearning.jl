# Tips for Developers

This page aims to provide a short introduction to Julia and some related packages for those who are interested in making contributions to this package.


## Basic

Go through the [latest doc](https://docs.julialang.org/en/latest/). Especially keep an eye on the following sections if you come from other programming languages:

- [Noteworthy Differences from other Languages](https://docs.julialang.org/en/latest/manual/noteworthy-differences/).

    You don't need to remember all those details, but giving a glimp of them will make things easier to transfer to Julia.

- [Function like objects](https://docs.julialang.org/en/latest/manual/methods/#Function-like-objects-1)

    It shouldn't surprise you that many important components in this package are functional objects.

- [Parametric Types](https://docs.julialang.org/en/v1/manual/types/#Parametric-Types-1)

    It may be a little struggling to create a flexible type if you have little background in parameteric types. See also [Design Patterns with Parameteric Methods](https://docs.julialang.org/en/latest/manual/methods/#Design-Patterns-with-Parametric-Methods-1)

- Traits and Multiple Dispatch

    You may have heard a lot of Julia users talking about Traits and Multiple Dispatch. This [blog](https://white.ucc.asn.au/2018/10/03/Dispatch,-Traits-and-Metaprogramming-Over-Reflection.html) provides a general introduction to them.

- [SubArrays](https://docs.julialang.org/en/latest/devdocs/subarrays/)

    To avoid unnecessary memory allocation, [views](https://docs.julialang.org/en/latest/base/arrays/#Views-(SubArrays-and-other-view-types)-1) are widely used in this package.

- [Performance Tips](https://docs.julialang.org/en/latest/manual/performance-tips/)

    Always keep those suggestions in mind if you want to make your algorithms run fast.

- [Multi-Threading](https://docs.julialang.org/en/latest/base/multi-threading/)

    Multi-Threading *will be* heavily use in some async algorithms.

## Code Structure

```
src
├── components
│   ├── action_selectors
│   ├── agents
│   ├── approximators
│   ├── buffers
│   ├── environment_models
│   ├── learners
│   └── policies
├── extensions
├── glue
└── Utils
```

- `components`: many important concepts in reinforcement learning have a corresponding component here.
- `extensions`: patches for upstream packages.
- `glue`: functions and types related to running experiments are here (like how to configure experiments, logging...).
- `Utils`: an inner module which provides many common helper functions.

## Some Important Packages

### [ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl)

ReinforcementLearningEnvironments.jl provides only some very common environments (for example: CartPoleEnv). But it also provides a set of unified interfaces for some other environments with the help of [Requires.jl](https://github.com/MikeInnes/Requires.jl). 

### [Flux.jl](https://github.com/FluxML/Flux.jl)

Flux.jl is a very lightweight package for deep learning. Its source code is really worth reading.

### [Zygote.jl](https://github.com/FluxML/Zygote.jl)

The next-gen AD system for Flux. Basically you only need to understand how to implement a customized `@adjoin`.

### [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl)

A basic understanding of how CuArrays.jl works will help you to write some efficient customized GPU kernels. [An Introduction to GPU Programming in Julia ](https://nextjournal.com/sdanisch/julia-gpu-programming) is also a very helpful resource.

### [Knet.jl](https://github.com/denizyuret/Knet.jl)

Another excellent julia package for deep learning. You'll find it easy to write algorithms that run on both Flux and Knet in this package.

## Q&A

*Your questions will appear here.*