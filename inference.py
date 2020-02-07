import cv2
from openvino.inference_engine import IENetwork, IECore

cpu_ext ="C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll"

def load_to_IE(model):
    # Getting the *.bin file location
    model_bin = model[:-3]+"bin"
    # Loading the Inference Engine API
    ie = IECore()
    
    # Loading IR files    
    net = IENetwork(model=model, weights = model_bin)
    
    # Checking if CPU extension is needed
    cpu_extension_needed = False
    network_layers = net.layers.keys()
    supported_layer_map = ie.query_network(network=net, device_name="CPU")
    supported_layers = supported_layer_map.keys()
    
    for layer in network_layers:
        if layer in supported_layers:
            pass
        else:
            cpu_extension_needed =True
            print("CPU extension needed")
            break
    
    # Adding CPU extension if needed
    if cpu_extension_needed:
        ie.add_extension(extension_path=cpu_ext, device_name="CPU")
        print("CPU extension added")
    else:
        print("CPU extension not needed")
    
    # Checking for any unsupported layers, if yes, exit
    supported_layer_map = ie.query_network(network=net, device_name="CPU")
    supported_layers = supported_layer_map.keys()
    unsupported_layer_exists = False
    network_layers = net.layers.keys()
    for layer in network_layers:
        if layer in supported_layers:
            pass
        else:
            print(layer +' : Still Unsupported')
            unsupported_layer_exists = True
    if unsupported_layer_exists:
        print("Exiting the program.")
        exit(1)
    
    # Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU")
    print("IR successfully loaded into Inference Engine.")

    return exec_net


def get_input_shape(model):
    """GIven a model, returns its input shape"""
    model_bin = model[:-3]+"bin"
    net = IENetwork(model=model, weights = model_bin)
    input_blob = next(iter(net.inputs))
    return net.inputs[input_blob].shape

def sync_inference(exec_net, image):
    input_blob = next(iter(exec_net.inputs))
    return exec_net.infer({input_blob: image})

def async_inference(exec_net, image, request_id=0):
    input_blob = next(iter(exec_net.inputs))
    exec_net.start_async(request_id, inputs={input_blob: image})
    return exec_net

def get_async_output(exec_net, request_id=0):
    output_blob = next(iter(exec_net.outputs))
    status = exec_net.requests[request_id].wait(-1)
    if status == 0:
        result = exec_net.requests[request_id].outputs[output_blob]
        return result

def preprocessing(input_image, height, width):
    """
    Given an image and desired height and width, 
    reshapes the image to desired height and width
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
    