FROM pytorch/pytorch:latest
RUN apt-get update \
    && apt-get install -y curl

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh

RUN pip install transformer-lens plotly einops protobuf==3.20.* jaxtyping==0.2.13 torchtyping jupyterlab scikit-learn ipywidgets matplotlib kaleido openai
RUN pip install typeguard==2.13.3
RUN pip install circuitsvis

# RUN git config --global user.email "oskar.hollinsworth@gmail.com" && \
#     git config --global user.name "skar0" && \
#     git config --global remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"

# Node.js # https://github.com/nodesource/distributions
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install -y ca-certificates curl gnupg
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    apt-get install -y nodejs

# Yarn
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y yarn

RUN pip install -U kaleido

RUN apt install wget -y
RUN wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.focal_amd64.deb
RUN apt install -f ./wkhtmltox_0.12.6-1.focal_amd64.deb -y
RUN pip install imgkit