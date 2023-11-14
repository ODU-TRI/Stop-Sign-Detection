# Stop Sign Detection Using YOLO and Automatic Geotag retrieval

Summary: The program attached to this summary report detects stop signs in an image and extracts geographical coordinates of the image from its metadata, if available. This code uses the YOLO8 library for stop detection and PIL for extracting geolocation data of the image.

Inputs and Outputs: Input to the code is a sequence of images stored in the ‘image’ folder. The code processes all images and summarizes the results in a csv file ‘stop_detection_result.csv’ which contains image names, confidence Score in detecting stop signs, Latitudes, and Longitudes. Additionally, the folder ‘stop_detection_result_images’ contains the original images with a box around all detected objects. YOLO8 pre-trained model detects 79 different objects. The traffic-related objects detectable by the pre-trained model include vehicles, stop signs, traffic lights, parking meters, and fire hydrants.

# Installation and Usage

[This google colab page](link-colab) offers step by step instruction on how to run the program.

**Step 1**: Loading required libraries
To run the code, we need the following libraries. Note that, library ultralytics is not installed on a Colab notebook by default. Therefore, we need to first install it onto the notebook using pip.

```python
!pip install ultralytics
from ultralytics import YOLO
```

**Step 2**: Loading Input Images:
Using a Colab environment, we can upload the images on Google Drive and load them into the code. To do this, the command drive.mount('/content/drive') loads the google drive. We can store the input images in a path like /Stop_Detection/images/. The library glob can look up for the files inside this folder and store file names and their addresses in a Python list as follows:

```python
drive.mount('/content/drive')
images = glob('/content/drive/MyDrive/Stop_Detection/images/*') # this line creates a list of file names in the folder 'images' #
print (f'{len(images)} images loaded successfully.')
```

**Step 3**: Run the YOLO model 
The pretrained YOLO8 is then loaded and we perform the detection for the list of images we have. This may take a few minutes.

```python
model = YOLO('yolov8m.pt') # this line loads YOLO pre-trained model #
results = model(images) # this line performs detection on the images #
```

**Step 4**: Formatting the Output and Extracting the Geotag Information

The pre-trained model detects 79 different objects, each labeled from 1 to 79. The ID for the stop sign is 11. The line conf_list = r.boxes.conf.numpy() extracts the confidence score of the detected objects. If no stop sign is detected the confidence score is set to 0. The lines after the "If statement extract" latitude and longitude from the image's metadata. If the location is not recorded, latitude and longitude are set to 999. The next lines append the file name, conf, lat, and lon of the processed image to the output list. Then we save the output image, with a bounding box around stop signs.

```python
result_list=[]          # creates an empty list for the output file #
result_list.append(['File Name', 'Stop Sign Confidence Score', 'Latitude', 'Longitude'])  # this line creates header line for the output file #
for i,r in enumerate(results): # a for loop to process each result #
    class_list = r.boxes.cls.numpy() # this line extracts the ID of the detected objects. The pretrained model detects 90 different objects each labled from 1 to 90. The ID for stop sign object is 11.#
    conf_list = r.boxes.conf.numpy() # this line extracts the confidence score of the detected objects #
    #### if no stop sign is detected the confidence score is set to 0 ####
    indices= np.where(class_list==11)[0]
    if indices.shape == (0,):
        conf = 0
    else:
        conf = np.max(conf_list[indices])
    ### these lines extract lat and lon from the image's meta data. If the location is not recorded, lat and lon are set to 999 ###
    try:
        img=Image.open(images[i])
        img._getexif()
        exif = {ExifTags.TAGS[k]: v for k,v in img._getexif().items()}
        lat =  exif['GPSInfo'][2][0]+exif['GPSInfo'][2][1]/60.+exif['GPSInfo'][2][2]/3600.
        lon =  exif['GPSInfo'][4][0]+exif['GPSInfo'][4][1]/60.+exif['GPSInfo'][4][2]/3600.
        lon = -lon
    except:
        lat=999
        lon=999
    # these lines add file name, conf, lat, lon of the proccessed image to the output list
    file_name = images[i].split('/')[6]
    result_list.append([file_name,f'{conf:.3f}',f'{lat:.5f}',f'{lon:.5f}'])
    # these lines save the output image in folder 'stop_detection_result_images'
    im_array = r[class_list == 11].plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.save(f'/content/drive/MyDrive/Stop_Detection/stop_detection_result_images/result_{file_name}')
    ##### end of For loop ##########
```

**Step 5**: Saving the output file

We can simply save the list of results after the loop using the following lines:

```python
df = pd.DataFrame(result_list)
df.to_csv("/content/drive/MyDrive/Stop_Detection/stop_detection_result.csv", mode='w', index=False)
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file



