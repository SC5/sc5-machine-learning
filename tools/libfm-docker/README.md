# libFM Docker
This project provides a dockerised version of the [libFM](http://www.libfm.org) Factorization Machine library.
From the website:

    Factorization machines (FM) are a generic approach that allows to mimic most factorization models by feature engineering. This way, factorization machines combine the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain.

## Building the image
To build, `cd` into the project root directory and run `docker build -t libfm .`

## Running the container
 To run the container, execute:

    docker run -it libfm /bin/bash

After running the above command, you should find yourself in the directory in which `libFM` was installed. The `libFM` executables can be found in the `bin/` subdirectory. For more information on how to use `libFM`, please consult the author's excellent [manual](http://www.libfm.org/libfm-1.42.manual.pdf) (PDF). 