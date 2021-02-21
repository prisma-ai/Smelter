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
            if self.int64Data.count == 0 {
                let array: [Int64] = self.rawData.array(length: self.length)
                return array.map(Int.init)
            }
            
            return self.int64Data.map(Int.init)
        case DataType.uint32.rawValue,
             DataType.uint64.rawValue:
            return self.uint64Data.map(Int.init)
        case DataType.float.rawValue:
            if self.floatData.count == 0 {
                let floats: [Float] = self.rawData.array(length: self.length)
                return floats.map(Int.init)
            }
            return self.floatData.map(Int.init)
        case DataType.double.rawValue:
            return self.doubleData.map(Int.init)
        case DataType.float16.rawValue:
            return self.rawData.convertingFloat16toFloat32(count: self.length).map(Int.init)
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
                let floats: [Float] = self.rawData.array(length: self.length)
                return floats
            }
            return self.floatData
        case DataType.double.rawValue:
            return self.doubleData.map(Float.init)
        case DataType.float16.rawValue:
            let count = self.rawData.count / MemoryLayout<Float16>.stride
            return self.rawData.convertingFloat16toFloat32(count: count)
        default:
             fatalError("Unsupported conversion rule")
        }
    }

    var length: Int {
        return Int(self.dims.reduce(1, *))
    }

}
