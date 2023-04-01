# Multi-Well Tools #
### Tools to extract data produced by multi-well microscope setups ###

### Basic System Test ###
To run a basic system test, ensure you are in the main directory i.e. multiwell_tools, then run:</br></br>
` > python ./src/runner.py`</br></br>
This will run the main signal extraction function on a test video,
produce xlsx files with the time series signal from each well and a zip of all the xlsx files,
an image with the roi's drawn on one frame of the input (video), and
plots of each wells time series signal.
