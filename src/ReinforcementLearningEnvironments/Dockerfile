FROM julia:1.3

# install dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        # ArcadeLearningEnvironment
        libz-dev \
        unzip \
        # ViZDoom
        wget \
        libboost-all-dev \
        build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
        nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
        libopenal-dev timidity libwildmidi-dev unzip \
        # PyCall OpenAI Gym
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools;

RUN ln -s /usr/bin/pip3 /usr/bin/pip; \
    ln -s /usr/bin/python3 /usr/bin/python; \
    pip install wheel gym;

ADD . /jl_pkg
WORKDIR /jl_pkg

# Following line can be removed after Hanabi.jl get registered.
RUN julia --color=yes -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/findmyway/Hanabi.jl", rev="master"))'
RUN julia --color=yes -e 'using Pkg; Pkg.clone(pwd()); Pkg.build("ReinforcementLearningEnvironments"; verbose=true); Pkg.test("ReinforcementLearningEnvironments"; coverage=true)'

CMD ["julia"]