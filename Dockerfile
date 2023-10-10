FROM pytorch/pytorch:latest
RUN pip install transformer-lens circuitsvis jaxtyping==0.2.13 einops protobuf==3.20.* plotly torchtyping jupyterlab scikit-learn ipywidgets matplotlib kaleido
RUN pip install git+https://github.com/neelnanda-io/neel-plotly.git
RUN pip install --upgrade jax jaxlib
RUN apt-get update \
    && apt-get install -y curl

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh

RUN pip install openai