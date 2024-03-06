FROM pytorch/pytorch:latest

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg wget magic-wormhole tzdata && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && apt-get install -y gh && \
    curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    apt-get update && apt-get install -y nodejs yarn && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY CircuitsVis CircuitsVis
RUN pip install --no-cache-dir -e ./CircuitsVis/python && \
    pip install --no-cache-dir git+https://github.com/neelnanda-io/neel-plotly.git && \
    pip install --no-cache-dir plotly einops protobuf==3.20.* jaxtyping==0.2.13 torchtyping jupyterlab scikit-learn ipywidgets matplotlib kaleido openai typeguard==2.13.3 kaleido==0.2.1 dill==0.3.4 imgkit jupyter ipykernel pytest pytest-doctestplus nbval pytest-cov jax==0.4.25 jaxlib==0.4.25 transformer_lens==1.14.0 hydra-core transformers_stream_generator accelerate tiktoken