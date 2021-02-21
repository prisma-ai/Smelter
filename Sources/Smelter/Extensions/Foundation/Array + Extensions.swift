import Foundation

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

