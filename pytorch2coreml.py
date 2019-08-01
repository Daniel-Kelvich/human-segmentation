from collections import OrderedDict
import yaml
import coremltools
import onnx
from onnx_coreml import convert
import torch.onnx
# from model_training.joint.models.unet_mobilenet import JointUNetMobileNet
# from model_training.head_seg_MHP_LIP_ATR_CIHP.models.unet_mobilenet import UNetMobileNet
# from model_training.head_seg_MHP_LIP_ATR_CIHP.models.fpn_mobilenet import FPNMobileNet
from src.unet_mobilenet import UNetMobileNet
import copy

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):
    """
    Convert an output multiarray to be represented as an image
    This will modify the Model_pb spec passed in.
    Example:
        model = coremltools.models.MLModel('MyNeuralNetwork.mlmodel')
        spec = model.get_spec()
        convert_multiarray_output_to_image(spec,'imageOutput',is_bgr=False)
        newModel = coremltools.models.MLModel(spec)
        newModel.save('MyNeuralNetworkWithImageOutput.mlmodel')
    Parameters
    ----------
    spec: Model_pb
        The specification containing the output feature to convert
    feature_name: str
        The name of the multiarray output feature you want to convert
    is_bgr: boolean
        If multiarray has 3 channels, set to True for RGB pixel order or false for BGR
    """
    for output in spec.description.output:
        if output.name != feature_name:
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':
            raise ValueError("%s is not a multiarray type" % output.name)
        array_shape = tuple(output.type.multiArrayType.shape)
        channels, height, width = array_shape
        from coremltools.proto import FeatureTypes_pb2 as ft
        if channels == 1:
            output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        elif channels == 3:
            if is_bgr:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('BGR')
            else:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')
        else:
            raise ValueError("Channel Value %d not supported for image inputs" % channels)
        output.type.imageType.width = width
        output.type.imageType.height = height


def pth_to_onnx(model, weights , onnx_name, w=256, h=256):
    dummy_input = torch.randn(1, 3, w, h).cuda()
    model.load_state_dict(weights)
    torch.onnx.export(model, dummy_input, onnx_name, verbose=True, export_params=True)


def onnx_to_coreml(config):
    model = onnx.load(config['onnx_name'])
    scales = [1.0 / (x * 255.0) for x in _STD]
    biases = [-(x * 255.0) for x in _MEAN]

    args = dict(is_bgr=False, red_bias=biases[0], green_bias=biases[1], blue_bias=biases[2])

    model = convert(model,
                    image_input_names=['0'],
                    preprocessing_args=args
                    )
    model.short_description = config['description']
    model = coremltools.models.MLModel(_add_preprocessing(model.get_spec(), scales))

    model.save(config['coreml_name'])
    if config['quantize']:
        quantized_model = coremltools.models.neural_network.quantization_utils.quantize_weights(
            model,
            nbits=16,
            quantization_mode='linear'
        )
        if not isinstance(quantized_model, coremltools.models.MLModel):
            quantized_model = coremltools.models.MLModel(quantized_model)
        quantized_model.save('quantize_{}'.format(config['coreml_name']))


def get_model(model_name, num_classes=1):
    # if model_name == 'fpn':
    #     return FPNMobileNet(output_ch=num_classes, pretrained=False).cuda()
    # elif model_name == 'joint_unet':
    #     return JointUNetMobileNet(num_classes=num_classes, pretrained=False).cuda()
    return UNetMobileNet(num_classes=num_classes, pretrained=False).cuda()


def _add_preprocessing(spec, scales):
    nn_spec = spec.neuralNetwork
    layers = nn_spec.layers  # this is a list of all the layers
    layers_copy = copy.deepcopy(layers)  # make a copy of the layers, these will be added back later
    del nn_spec.layers[:]  # delete all the layers

    # add a scale layer now
    # since mlmodel is in protobuf format, we can add proto messages directly
    # To look at more examples on how to add other layers: see "builder.py" file in coremltools repo
    scale_layer = nn_spec.layers.add()
    scale_layer.name = 'scale_layer'
    scale_layer.input.append('0')
    scale_layer.output.append('input1_scaled')

    params = scale_layer.scale
    params.scale.floatValue.extend(scales)  # scale values for RGB
    params.shapeScale.extend([3, 1, 1])  # shape of the scale vector

    # now add back the rest of the layers (which happens to be just one in this case: the crop layer)
    nn_spec.layers.extend(layers_copy)

    # need to also change the input of the crop layer to match the output of the scale layer
    nn_spec.layers[1].input[0] = 'input1_scaled'
    return spec


def _get_state_dict(config):
    data = torch.load(config['model_path'])
    # If model was saved with nn.DataParallel we need to remove module. extension frome state_dict
    state_dict = data['model']
    if config['parallel']:
        state_dict = OrderedDict()
        for k, v in data['model'].items():
            name = k[7:] if k.startwith('module.') else k
            state_dict[name] = v
    return state_dict


if __name__ == '__main__':
    with open('convert.yaml', 'r') as f:
        config = yaml.load(f)
    # state_dict = _get_state_dict(config)
    state_dict = torch.load(config['model_path'])
    pth_to_onnx(get_model(config['model'], config['num_classes']), state_dict, config['onnx_name'])
    onnx_to_coreml(config)
