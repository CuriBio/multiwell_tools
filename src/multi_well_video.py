import numpy as np
import cv2 as cv
import openpyxl
import zipfile
import matplotlib.pyplot as plt
from typing import Dict
from os import makedirs, walk as dir_ls
from os.path import isdir, basename, join as join_paths
from video_api import VideoReader


"""
    Extracts signals from multi-well microscope data and stores each wells signal data as a separate xlsx file.
    Creates an image with all the roi's drawn on one frame & creates an image with signal plots for each well.
"""


def well_signals(setup_config: Dict):
    """
        Extracts time series data for defined roi's within each frame of a video and
        writes the results to xlsx files for each roi and zips them into a single archive.
        Also creates an image with all the roi's drawn on one frame as a quick sanity check,
        and creates another image with time series plots of the signal data for each well.
    """

    # add the roi_coordinates to the well information
    setup_config = add_roi_coordinates_to_well_info(setup_config)

    # safely create the output dir if it does not exist
    print("\nCreating Output Dir")
    make_output_dir(setup_config['output_dir_path'])

    # open the input stream
    print("\nOpening Video")
    input_video_stream = VideoReader(setup_config['input_path'])
    if not input_video_stream.isOpened():
        raise RuntimeError("Can't open video stream. No Ca2+ signals have been extracted.")

    # save an image with the roi's drawn on it as quick sanity check.
    print("\nCreating ROI Sanity Check Image...")
    input_video_stream.setFramePosition(0)
    frame_to_draw_rois_on = input_video_stream.frameRGB()
    path_to_save_frame_image = join_paths(setup_config['output_dir_path'], 'roi_locations.png')
    frame_with_rois_drawn(frame_to_draw_rois_on, setup_config['wells'], path_to_save_frame_image)
    print("ROI Sanity Check Image Created")

    # create a numpy array to store the time series of well signal values
    num_wells = num_active_wells(setup_config['wells'])
    num_frames = input_video_stream.numFrames()
    signal_values = np.empty((num_wells, num_frames), dtype=np.float32)

    # extract ca2+ signal in each well for each frame
    print("\nStarting Signal Extraction...")
    input_video_stream.initialiseStream()
    while input_video_stream.next():
        # print(f"processing frame number: {input_video_stream.framePosition()} of {num_frames}")
        for well_name, well_info in setup_config['wells'].items():
            if well_info['is_active']:
                x_start = int(well_info['roi_coordinates']['upper_left']['x_pos'])
                x_end = int(well_info['roi_coordinates']['lower_right']['x_pos'])
                y_start = int(well_info['roi_coordinates']['upper_left']['y_pos'])
                y_end = int(well_info['roi_coordinates']['lower_right']['y_pos'])
                frame_roi = input_video_stream.frameGray()[y_start:y_end, x_start:x_end]
                row_num = well_info['serial_position']
                frame_num = input_video_stream.framePosition()
                signal_values[row_num, frame_num] = roi_signal(frame_roi)
    print("Signal Extraction Complete")

    # write each roi's time series data to an xlsx file
    print("\nWriting ROI Signals to XLSX files...")
    if 'duration' not in setup_config:
        setup_config['duration'] = input_video_stream.duration()
    if 'fps' not in setup_config:
        setup_config['fps'] = input_video_stream.framesPerSecond()
    time_stamps = np.linspace(start=0, stop=setup_config['duration'], num=num_frames)
    setup_config['xlsx_output_dir_path'] = join_paths(setup_config['output_dir_path'], 'xlsx')
    make_xlsx_output_dir(xlsx_output_dir_path=setup_config['xlsx_output_dir_path'])
    signal_to_xlsx_for_sdk(signal_values, time_stamps, setup_config)
    print("Writing Signals to XLSX Files Complete")

    # zip all the xlsx files into a single archive
    print("\nCreating Zip Archive For XLSX files...")
    xlsx_archive_file_path = join_paths(setup_config['output_dir_path'], 'xlsx-results.zip')
    zip_files(input_dir_path=setup_config['xlsx_output_dir_path'], zip_file_path=xlsx_archive_file_path)
    print("Zip Archive For XLSX files Created")

    # save an image with a plot of all well signals
    print("\nCreating Signal Plot Sanity Check Image...")
    setup_config['num_well_rows'] = 0
    setup_config['num_well_cols'] = 0
    for well_name, well_info in setup_config['wells'].items():
        well_grid_position = well_info['grid_position']
        if well_grid_position['row'] > setup_config['num_well_rows']:
            setup_config['num_well_rows'] = well_grid_position['row']
        if well_grid_position['col'] > setup_config['num_well_cols']:
            setup_config['num_well_cols'] = well_grid_position['col']
    # grid rows and columns are 0 indexed so need to increment by 1 to get correct number
    setup_config['num_well_rows'] += 1
    setup_config['num_well_cols'] += 1
    plot_file_path = join_paths(setup_config['output_dir_path'], 'roi_signals_plots.png')
    signals_to_plot(signal_values, time_stamps, setup_config, plot_file_path)
    print("Signal Plot Sanity Check Image Created")

    # clean up
    input_video_stream.close()
    print()


def signals_to_plot(signal_values: np.ndarray, time_stamps: np.ndarray, setup_config: Dict, plot_file_path: str):
    """ Create an image with plots of time series data for multiple ROIs """

    fig, axes = plt.subplots(
        nrows=setup_config['num_well_rows'], ncols=setup_config['num_well_cols'],
        figsize=(setup_config['num_well_cols']*3, setup_config['num_well_rows']*3),
        dpi=300.0,
        layout='constrained'
    )
    instrument_name = setup_config['instrument_name']
    recording_date = setup_config['recording_date']
    video_file_name = basename(setup_config['input_path'])
    plot_title = f"{instrument_name} Experiment Data - {recording_date} - {video_file_name}"
    fig.suptitle(plot_title, fontsize=20)
    fig.supylabel('ROI Average')
    fig.supxlabel('Time (s)')

    for well_name, well_info in setup_config['wells'].items():
        serial_position = well_info['serial_position']
        well_signal = signal_values[serial_position, :]
        plot_row = well_info['grid_position']['row']
        plot_col = well_info['grid_position']['col']
        axes[plot_row, plot_col].plot(time_stamps, well_signal)
        axes[plot_row, plot_col].set_title(well_name)

    plt.savefig(plot_file_path)


def zip_files(input_dir_path: str, zip_file_path: str):
    zip_file = zipfile.ZipFile(zip_file_path, 'w')
    for dir_name, _, file_names in dir_ls(input_dir_path):
        for file_name in file_names:
            file_path = join_paths(dir_name, file_name)
            zip_file.write(file_path, basename(file_path))
    zip_file.close()


def signal_to_xlsx_for_sdk(signal_values: np.ndarray, time_stamps: np.ndarray, setup_config: Dict):
    """ writes time series data to xlsx files for multiple ROIs """

    num_wells, num_data_points = signal_values.shape
    frames_per_second = setup_config['fps']
    date_stamp = setup_config['recording_date'] + ' 00:00:00'
    output_dir = setup_config['xlsx_output_dir_path']

    if 'barcode' in setup_config:
        well_plate_barcode = setup_config['barcode']
    else:
        well_plate_barcode = 'NA'

    for well_name, well_info in setup_config['wells'].items():
        workbook = openpyxl.Workbook()
        sheet = workbook.active

        # add meta data
        sheet['E2'] = well_name
        sheet['E3'] = date_stamp
        sheet['E4'] = well_plate_barcode
        sheet['E5'] = frames_per_second
        sheet['E6'] = 'y'  # do twitch's point up
        sheet['E7'] = 'NAUTILUS'  # microscope name

        # add runtime data (time, displacement etc)
        template_start_row = 2
        time_column = 'A'
        signal_column = 'B'
        well_data_row = well_info['serial_position']
        for data_point_position in range(num_data_points):
            sheet_row = str(data_point_position + template_start_row)
            sheet[time_column + sheet_row] = time_stamps[data_point_position]
            sheet[signal_column + sheet_row] = signal_values[well_data_row, data_point_position]
        path_to_output_file = join_paths(output_dir, well_name + '.xlsx')
        workbook.save(filename=path_to_output_file)
        workbook.close()


def frame_with_rois_drawn(frame_to_draw_on: np.ndarray, wells_info: Dict, path_to_save_frame_image: str):
    """ Draw multiple ROIs on one frame image """
    green_line_colour_bgr = (0, 255, 0)
    for _, well_info in wells_info.items():
        top_left = (
            int(well_info['roi_coordinates']['upper_left']['x_pos']),
            int(well_info['roi_coordinates']['upper_left']['y_pos'])
        )
        lower_right = (
            int(well_info['roi_coordinates']['lower_right']['x_pos']),
            int(well_info['roi_coordinates']['lower_right']['y_pos']),
        )
        cv.rectangle(
            img=frame_to_draw_on,
            pt1=top_left,
            pt2=lower_right,
            color=green_line_colour_bgr,
            thickness=1,
            lineType=cv.LINE_AA
        )
    cv.imwrite(path_to_save_frame_image, frame_to_draw_on)


def num_active_wells(wells: Dict) -> int:
    """ returns the count of wells marked as active """
    active_well_count = 0
    for _, well_info in wells.items():
        if well_info['is_active']:
            active_well_count += 1
    return active_well_count


def well_roi_coordinates(well_info: Dict, roi_info: Dict, scale_factor: float) -> Dict:
    """ returns a dictionary with coordinates of the roi """

    # NOTE: initially we will only return a 2D rectangular roi
    #       we also don't check for going outside the image

    roi_x_radius = roi_info['width']/2.0
    roi_y_radius = roi_info['height']/2.0
    # NOTE: images are stored "upside down", so upper visually is lower in coordinates,
    #       hence the minus y radius for upper and plus y radius for lower
    return {
        'upper_left': {
            'x_pos': (well_info['center_coordinates']['x_pos'] - roi_x_radius)/scale_factor,
            'y_pos': (well_info['center_coordinates']['y_pos'] - roi_y_radius)/scale_factor
        },
        'upper_right': {
            'x_pos': (well_info['center_coordinates']['x_pos'] + roi_x_radius)/scale_factor,
            'y_pos': (well_info['center_coordinates']['y_pos'] - roi_y_radius)/scale_factor
        },
        'lower_left': {
            'x_pos': (well_info['center_coordinates']['x_pos'] - roi_x_radius)/scale_factor,
            'y_pos': (well_info['center_coordinates']['y_pos'] + roi_y_radius)/scale_factor
        },
        'lower_right': {
            'x_pos': (well_info['center_coordinates']['x_pos'] + roi_x_radius)/scale_factor,
            'y_pos': (well_info['center_coordinates']['y_pos'] + roi_y_radius)/scale_factor
        }
    }


def add_roi_coordinates_to_well_info(setup_config: Dict) -> Dict:
    new_setup_config = setup_config.copy()
    scale_factor = new_setup_config['scale_factor']
    for _, well_info in new_setup_config['wells'].items():
        well_info['roi_coordinates'] = well_roi_coordinates(well_info, setup_config['roi'], scale_factor)
    return new_setup_config


def make_output_dir(output_dir_path: str):
    """ create the main output dir """
    if not isdir(output_dir_path):
        makedirs(name=output_dir_path, exist_ok=False)


def make_xlsx_output_dir(xlsx_output_dir_path: str):
    """ create a subdir for xlsx files """
    if not isdir(xlsx_output_dir_path):
        makedirs(name=xlsx_output_dir_path, exist_ok=False)


def roi_signal(roi_with_signal: np.ndarray) -> float:
    return np.mean(roi_with_signal)
