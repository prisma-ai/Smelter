//
//  NodeConverter.swift
//  Smelter
//
//  Created by Andrey Volodin on 16/04/2019.
//

import MetalPerformanceShaders

internal protocol NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws
}
