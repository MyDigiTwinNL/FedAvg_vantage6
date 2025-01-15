# basic python3 image as base
FROM --platform=linux/amd64  harbor2.vantage6.ai/infrastructure/algorithm-base

# This is a placeholder that should be overloaded by invoking
# docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="federated_cvdm_training_poc"

# install federated algorithm
COPY . /app
#RUN pip install     /app
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu /app


# Set environment variable to make name of the package available within the
# docker image.
ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `wrap_algorithm()` when the image is run. This function
# will ensure that the algorithm method is called properly.
CMD python -c "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"