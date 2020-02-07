import cv2
from inference import preprocessing, load_to_IE, sync_inference, async_inference, get_input_shape, get_async_output

def main():
    model = "model/intel/face-detection-adas-0001/INT8/face-detection-adas-0001.xml"    
    image = "image.jpg"
    
    image = cv2.imread(image)
    exec_net = load_to_IE(model)
    
    n, c, h, w = get_input_shape(model)
    preprocessed_image = preprocessing(image, h, w)
    
    result = sync_inference(exec_net, image = preprocessed_image)
    #print(result.keys())
    result = result['detection_out']
    output = output_processing(result, image)
    cv2.imwrite("output.png", output)
    
    # async
    async_net = async_inference(exec_net, image = preprocessed_image)
    async_result = get_async_output(async_net, request_id=0)
    
    async_output = output_processing(async_result, image)
    cv2.imwrite("async output.png", async_output)

def output_processing(result, image, threshold = 0.5):    
    color = (0,0,255)
    width = image.shape[1]
    height = image.shape[0]

    for i in range(len(result[0][0])): # Output shape is 1x1xNx7
        box = result[0][0][i]
        conf = box[2]       
        if conf >= threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
    return image

if __name__ == "__main__":
    main()
