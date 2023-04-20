# Multi-Well Tools #
## Tools to extract data produced by multi-well microscope experiments i.e. Nautilus ##

### Run the script from the command line ###
` > python /path/to/well_data.py /path/to/json_config --input_video_path /path/to/input/video --output_dir_path /path/to/output/dir`</br></br>
args with a leading -- are optional and if supplied will override keys with the same name in the json config.

### Basic System Test ###
To run a basic system test, if you are in the main directory (multiwell_tools) :</br></br>
python ./src/well_data.py ./test_data/Curi24wPconfig.json --input_video_path .\test_data\PBS2023_04_05_004446.raw --output_dir_path .\test_data\PBS2023_04_05_004446Results --num_horizontal_pixels 1536 --num_vertical_pixels 1024 --num_frames 15000 --bit_depth 8 --scale_factor 2 --duration 60 --fps 250
python ./src/well_data.py ./test_data/Curi24wPconfig.json --input_video_path .\test_data\Curi24wP.raw --output_dir_path .\test_data\Curi24wPResults --num_horizontal_pixels 1536 --num_vertical_pixels 1024 --num_frames 250 --bit_depth 8 --scale_factor 2 --duration 1 --fps 250
python ./src/well_data.py ./test_data/Costar96wPconfig.json --input_video_path .\test_data\Costar96wP.raw --output_dir_path .\test_data\Costar96wPResults --num_horizontal_pixels 1536 --num_vertical_pixels 1024 --num_frames 1250 --bit_depth 8 --scale_factor 2 --duration 5 --fps 250
python ./src/well_data.py ./test_data/Costar96wPconfig.json --input_video_path .\test_data\Costar96wP16bit.raw --output_dir_path .\test_data\Costar96w16bitPResults --num_horizontal_pixels 1536 --num_vertical_pixels 1024 --num_frames 1250 --bit_depth 16 --scale_factor 2 --duration 5 --fps 250
This will run the well data extraction function which will produce:
* xlsx files with the time series data from each well,
* a zip archive of all the xlsx files,
* an image with the roi's drawn on one frame of the input video,
* a plot of every wells time series data.

