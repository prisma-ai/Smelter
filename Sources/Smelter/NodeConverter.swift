import MetalPerformanceShaders

internal protocol NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws
}
