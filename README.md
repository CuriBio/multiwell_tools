# Multi-well Tools #
### Tools to extract data produced by multi well microscope setups ###

### Basic System Test ###
To run a basic system test, from the main directory run:</br>
` > python -m test.system_test`</br>
This will run the main signal extraction function on a test video, 
produce xlsx files with the time series signal from each well and a zip of all the xlsx files,
an image with the roi's drawn on one frame of the input (video), and
plots of each wells time series signal.
