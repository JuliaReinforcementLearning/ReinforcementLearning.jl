# Frequently Asked Questions

## I get an error when trying examples in the doc

This documentation is generated with the following environment setup. If you
get any unexpected exception or find any inconsistency between the result on
your machine and the documentation. Please [file an
issue](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues)
with the following message to help us improve the doc.

```@setup versions
using InteractiveUtils
```

```@repl versions
using Pkg, Dates
today()
versioninfo()
buff = IOBuffer();Pkg.status(io=buff);println(String(take!(buff)))
```

## Downgrade happens when using this package

This may happen occasionally. The reason is complex, either because we haven't
updated the compat section yet, or some other packages you are using relies on
an old dependency. Pleas [create an
issue](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues)
to describe how it happens and we can work on it together to resolve it.

