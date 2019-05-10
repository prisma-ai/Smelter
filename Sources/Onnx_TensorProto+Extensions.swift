//
//  Onnx_TensorProto+Extensions.swift
//  Smelter
//
//  Created by Eugene Bokhan on 06/05/2019.
//

import Alloy

extension Onnx_TensorProto {

    var integers: [Int] {
        switch Int(self.dataType) {
        case DataType.int32.rawValue,
             DataType.int16.rawValue,
             DataType.int8.rawValue,
             DataType.uint16.rawValue,
             DataType.uint8.rawValue,
             DataType.bool.rawValue:
            return self.int32Data.map(Int.init)
        case DataType.int64.rawValue:
            return self.int64Data.map(Int.init)
        case DataType.uint32.rawValue,
             DataType.uint64.rawValue:
            return self.uint64Data.map(Int.init)
        case DataType.float.rawValue:
            if self.floatData.count == 0 {
                return self.rawData.withUnsafeBytes { (p: UnsafePointer<Float>) in
                    return Array(UnsafeBufferPointer<Float>(start: p,
                                                            count: self.length))
                }.map(Int.init)
            }
            return self.floatData.map(Int.init)
        case DataType.double.rawValue:
            return self.doubleData.map(Int.init)
        case DataType.float16.rawValue:
            return (self.rawData.withUnsafeBytes {
                float16to32(UnsafeMutableRawPointer(mutating: $0),
                            count: self.length)
            } ?? []).map(Int.init)
        default:
            fatalError("Unsupported conversion rule")
        }
    }

    var floats: [Float] {
        switch Int(self.dataType) {
        case DataType.int32.rawValue,
             DataType.int16.rawValue,
             DataType.int8.rawValue,
             DataType.uint16.rawValue,
             DataType.uint8.rawValue,
             DataType.bool.rawValue:
            return self.int32Data.map(Float.init)
        case DataType.int64.rawValue:
            return self.int64Data.map(Float.init)
        case DataType.uint32.rawValue,
             DataType.uint64.rawValue:
            return self.uint64Data.map(Float.init)
        case DataType.float.rawValue:
            if self.floatData.count == 0 {
                return self.rawData.withUnsafeBytes { (p: UnsafePointer<Float>) in
                    return Array(UnsafeBufferPointer<Float>(start: p,
                                                            count: self.length))
                }
            }
            return self.floatData
        case DataType.double.rawValue:
            return self.doubleData.map(Float.init)
        case DataType.float16.rawValue:
            let count = self.rawData.count / MemoryLayout<Float16>.stride
            return self.rawData.withUnsafeBytes {
                float16to32(UnsafeMutableRawPointer(mutating: $0),
                            count: count)
            } ?? []
        default:
             fatalError("Unsupported conversion rule")
        }
    }

    var length: Int {
        return Int(self.dims.reduce(1, *))
    }

}
