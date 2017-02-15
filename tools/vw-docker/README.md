# Vowpal Wabbit Docker
This project provides a dockerised (and daemonised) version of [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki) (VW), a fast out-of-core machine learning library developed at Yahoo and Microsoft.

## Building the image
To build, `cd` into the project root directory and run `docker build -t vw .`

## Running the container
To run, execute `docker run -p 26542:26542 -d vw` in the root directory of the project. The container will expose the VW daemon at port 26542, with the VW option `--cb-explore`. You can also attach using `docker exec` and run VW directly inside the container.

## Example use
Assuming the container is running with the default options, you can get predictions like so:

`echo " | a b c" | netcat localhost 26542`

And fit training examples like so:

`echo "1:1:0.5 | a b c" | netcat localhost 26542`

For more information on VW's input format, see [here](https://github.com/JohnLangford/vowpal_wabbit/wiki/Tutorial).

## Configuration
VW supports a number of algorithms and command line options, see [here](https://github.com/JohnLangford/vowpal_wabbit/wiki/Command-line-arguments) for more information. The same options can be used in daemon mode, just edit `entrypoint.sh` to your liking.