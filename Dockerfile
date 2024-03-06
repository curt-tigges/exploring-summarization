FROM pytorch/pytorch:latest
RUN apt-get update \
    && apt-get install -y curl ca-certificates curl gnupg wget magic-wormhole
# github cli
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
# nodejs
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    apt-get install -y nodejs
# Yarn
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y nodejs && \
    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y yarn
# pip
COPY CircuitsVis CircuitsVis
RUN pip install -e ./CircuitsVis/python && \
    pip install plotly einops protobuf==3.20.* jaxtyping==0.2.13 torchtyping jupyterlab scikit-learn ipywidgets matplotlib kaleido openai typeguard==2.13.3 kaleido==0.2.1 dill==0.3.4 imgkit jupyter ipykernel pytest pytest-doctestplus nbval pytest-cov jax==0.4.25 jaxlib==0.4.25 transformer_lens==1.14.0 hydra-core transformers_stream_generator accelerate tiktoken && \
    pip install git+https://github.com/neelnanda-io/neel-plotly.git