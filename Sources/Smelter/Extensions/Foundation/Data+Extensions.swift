import Alloy

extension Data {
    func array<T>() -> [T] {
        let count = self.count / MemoryLayout<T>.stride
        let result = self.withUnsafeBytes {
            $0.baseAddress.flatMap {
                $0.bindMemory(to: T.self, capacity: count)
            }.flatMap {
                Array(UnsafeBufferPointer<T>(start: $0, count: count))
            }
        } ?? []
        return result
    }

    func convertingFloat16toFloat32(count: Int) -> [Float] {
        let result = self.withUnsafeBytes {
            $0.baseAddress.flatMap {
                float16to32(UnsafeMutableRawPointer(mutating: $0), count: count)
            }
        } ?? []
        return result
    }
}
