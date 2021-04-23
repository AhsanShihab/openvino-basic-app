import cv2
from openvino.inference_engine import IENetwork, IECore

def load_model(model):
    # Getting the *.bin file location
    model_bin = model[:-3]+"bin"
    # Loading the Inference Engine API
    ie = IECore()
    
    # Loading IR files    
    net = ie.read_network(model=model, weights = model_bin)

    # Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU")
    print("IR successfully loaded into Inference Engine.")
    
    return exec_net


def get_input_shape(net):
    """returns input shape of the given network"""
    input_key = list(net.input_info.keys())[0]                  # the model we used has only one input. 
                                                                # If the model takes multiple inputs, do more appropriate processing
    input_shape = net.input_info[input_key].input_data.shape
    
    return input_shape

def sync_inference(exec_net, image):
    input_key = list(exec_net.input_info.keys())[0]
    output_key = list(exec_net.outputs.keys())[0]

    result = exec_net.infer({input_key: image})
    print('Sync inference successful')

    return result[output_key]

def async_inference(exec_net, image, request_id=0):
    input_key = list(exec_net.input_info.keys())[0]

    return exec_net.start_async(request_id, inputs={input_key: image})

def get_async_output(async_handler):  
    status = async_handler.wait()
    output_key = list(async_handler.output_blobs.keys())[0] 
    result = async_handler.output_blobs[output_key].buffer
    print('Async inference successful')
    return result

def preprocessing(input_image, height, width):
    """
    Given an image and desired height and width, 
    reshapes the image to that height and width
    and brings color channel infront
    """
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

def main():
    load_to_IE(model_xml, model_bin)


if __name__ == "__main__":
    load_to_IE(model_xml, model_bin)
    