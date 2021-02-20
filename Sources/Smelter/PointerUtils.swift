import Foundation
import Alloy

extension Data {
    func convertToArrayOfType<T>(_ type: T.Type, length: Int) -> [T] {
        let result = self.withUnsafeBytes {
            $0.baseAddress.flatMap {
                $0.bindMemory(to: T.self, capacity: length)
            }.flatMap {
                Array(UnsafeBufferPointer<T>(start: $0, count: length))
            }
        } ?? []
        return result
    }

    func convertF16toF32(count: Int) -> [Float] {
        let result = self.withUnsafeBytes {
            $0.baseAddress.flatMap {
                float16to32(UnsafeMutableRawPointer(mutating: $0), count: count)
            }
        } ?? []
        return result
    }
}

extension Array {
    var unsafeMutablePointer: UnsafeMutablePointer<Element>? {
        self.withUnsafeBufferPointer {
            return UnsafeMutablePointer(mutating: $0.baseAddress)
        }
    }

    var unsafeMutableRawPointer: UnsafeMutableRawPointer? {
        self.withUnsafeBufferPointer {
            return UnsafeMutableRawPointer(mutating: $0.baseAddress)
        }
    }
}
