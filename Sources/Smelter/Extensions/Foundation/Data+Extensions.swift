import Alloy

extension Data {
    func array<T>(length: Int) -> [T] {
        let result = self.withUnsafeBytes {
            $0.baseAddress.flatMap {
                $0.bindMemory(to: T.self, capacity: length)
            }.flatMap {
                Array(UnsafeBufferPointer<T>(start: $0, count: length))
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
