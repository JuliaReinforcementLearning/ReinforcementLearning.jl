ARG IMAGE=nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM $IMAGE

ENV JULIA_CUDA_USE_BINARYBUILDER=false

# julia

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install --yes --no-install-recommends \
                    # basic stuff
                    curl ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# NOTE: this extracts the Julia version (assumed major.minor.patch) from the
#       Project.toml to keep it in sync with the GitHub Action workflow.

RUN VERSION=1.5.3 && \
    RELEASE=$(echo $VERSION | cut -d '.' -f 1,2 ) && \
    curl -s -L https://julialang-s3.julialang.org/bin/linux/x64/${RELEASE}/julia-${VERSION}-linux-x86_64.tar.gz | \
    tar -C /usr/local -x -z --strip-components=1 -f -

COPY . /rl

RUN julia -e 'using Pkg; Pkg.dev(url="/rl"); Pkg.instantiate(); Pkg.API.precompile()'

ENTRYPOINT ["/usr/local/bin/julia"]