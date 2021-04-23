import cv2
from inference import preprocessing, load_model, sync_inference, async_inference, get_input_shape, get_async_output

def main():
    model = "model/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"    
    image = "image.jpg"
    
    image = cv2.imread(image)
    exec_net = load_model(model)
    
    n, c, h, w = get_input_shape(exec_net)
    preprocessed_image = preprocessing(image, h, w)
    
    # sync
    result_sync = sync_inference(exec_net, image = preprocessed_image)
    output_sync = output_processing(result_sync, image)
    cv2.imwrite("output_sync.png", output_sync)
    
    # async
    async_handler = async_inference(exec_net, image = preprocessed_image)
    result_async = get_async_output(async_handler)
    output_async = output_processing(result_async, image)
    cv2.imwrite("output_async.png", output_async)

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
