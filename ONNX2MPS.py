import onnx
from onnx import optimizer
from onnx import numpy_helper
import numpy as np
import argparse


DATA_TYPES = {
    np.float32: 1, 
    np.float16: 10
}


def tensorInfosToType(tensor_infos, data_type):
    return [
        onnx.helper.make_tensor_value_info(
            tensor_info.name,
            DATA_TYPES[data_type],
            [dim.dim_value for dim in tensor_info.type.tensor_type.shape.dim]
        ) for tensor_info in tensor_infos
    ]


def tensorsToType(tensors, data_type):
    return [
        numpy_helper.from_array(
            numpy_helper.to_array(tensor).astype(data_type),
            name=tensor.name
        )
        for tensor in tensors
    ]


def swizzle(lst, swizzle_lst):
    assert len(lst) == len(swizzle_lst)
    return [lst[idx] for idx in swizzle_lst]


def reformatConvWeightTensorInfos(tensor_infos, swizzle_plan):
    for idx in range(len(tensor_infos)):
        if tensor_infos[idx].name not in swizzle_plan:
            continue
        tensor_infos[idx] = onnx.helper.make_tensor_value_info(
            tensor_infos[idx].name,
            tensor_infos[idx].type.tensor_type.elem_type,
            swizzle(
                [dim.dim_value for dim in tensor_infos[idx].type.tensor_type.shape.dim],
                swizzle_plan[tensor_infos[idx].name]
            )
        )
    return tensor_infos


def reformatConvWeightTensors(tensors, swizzle_plan, is_transpose):
    for idx in range(len(tensors)):
        if tensors[idx].name not in swizzle_plan:
            continue
        array = numpy_helper.to_array(tensors[idx]).transpose(
            swizzle_plan[tensors[idx].name]
        )
        if is_transpose[tensors[idx].name]:
            array = array[:,::-1,::-1,:]
        tensors[idx] = numpy_helper.from_array(
            array,
            name=tensors[idx].name
        )
    return tensors


def modelToMPS(model, data_type):
    is_transpose = dict()
    swizzle_plan = dict()
    for node in model.graph.node:
        if node.op_type == 'Conv':
            swizzle_plan[node.input[1]] = [0, 2, 3, 1]
            is_transpose[node.input[1]] = False
        if node.op_type == 'ConvTranspose':
            swizzle_plan[node.input[1]] = [1, 2, 3, 0]
            is_transpose[node.input[1]] = True
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=model.graph.node,
            name=model.graph.name,
            inputs=reformatConvWeightTensorInfos(
                tensorInfosToType(model.graph.input, data_type),
                swizzle_plan
            ),
            outputs=reformatConvWeightTensorInfos(
                tensorInfosToType(model.graph.output, data_type),
                swizzle_plan
            ),
            initializer=reformatConvWeightTensors(
                tensorsToType(model.graph.initializer, data_type),
                swizzle_plan,
                is_transpose
            )
        ),
        producer_name='ONNX2MPS',
        producer_version='1.0.0'
    )
    return model


def optimize_model(model, data_type):
    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    model = optimizer.optimize(model, ['fuse_bn_into_conv'])
    model = modelToMPS(model, data_type)
    return model


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to MPSNNGraphONNXBuilder format')
    parser.add_argument('--half', required=False, help='Use FP16 weights', action='store_true')
    parser.add_argument('--input', required=True, help='Path to ONNX model')
    parser.add_argument('--output', required=True, help='Path to MPS model')
    args = parser.parse_args()
    print('Parsed args')
    print('half: {}'.format(args.half))
    print('input: {}'.format(args.input))
    print('output: {}'.format(args.output))
    print()

    print('Started convertion')

    data_type = np.float32
    if args.half:
    	data_type = np.float16


    onnx_model = onnx.load(args.input)
    mps_model = optimize_model(onnx_model, data_type)
    onnx.save(mps_model, args.output)
    print('Success')

if __name__ == "__main__":
    main()
