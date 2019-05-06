//
//  Onnx_TensorProto+Extensions.swift
//  Smelter
//
//  Created by Eugene Bokhan on 06/05/2019.
//

extension Onnx_TensorProto {

    var integers: [Int] {
        switch Int(self.dataType) {
        case DataType.int32.rawValue,
             DataType.int16.rawValue,
             DataType.int8.rawValue,
             DataType.uint16.rawValue,
             DataType.uint8.rawValue,
             DataType.bool.rawValue,
             DataType.float16.rawValue:
            return self.int32Data.map { Int($0) }
        case DataType.int64.rawValue:
            return self.int64Data.map { Int($0) }
        case DataType.uint32.rawValue,
             DataType.uint64.rawValue:
            return self.uint64Data.map { Int($0) }
        case DataType.float.rawValue,
             DataType.complex64.rawValue:
            return self.floatData.map { Int($0) }
        case DataType.double.rawValue,
             DataType.complex128.rawValue:
            return self.doubleData.map { Int($0) }
        default: return []
        }
    }

    var floats: [Float] {
        switch Int(self.dataType) {
        case DataType.int32.rawValue,
             DataType.int16.rawValue,
             DataType.int8.rawValue,
             DataType.uint16.rawValue,
             DataType.uint8.rawValue,
             DataType.bool.rawValue,
             DataType.float16.rawValue:
            return self.int32Data.map { Float($0) }
        case DataType.int64.rawValue:
            return self.int64Data.map { Float($0) }
        case DataType.uint32.rawValue,
             DataType.uint64.rawValue:
            return self.uint64Data.map { Float($0) }
        case DataType.float.rawValue,
             DataType.complex64.rawValue:
            return self.floatData
        case DataType.double.rawValue,
             DataType.complex128.rawValue:
            return self.doubleData.map { Float($0) }
        default: return []
        }
    }

}
