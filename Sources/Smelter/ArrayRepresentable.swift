import Foundation

protocol ArrayRepresentable {
    associatedtype T: Equatable
    static var keyPaths: [WritableKeyPath<Self, T>] { get }
    func toArray() -> [T]
    init()
    init(array: [T])
}

extension ArrayRepresentable {
    
    init(array: [T]) {
        let keyPaths = Self.keyPaths
        assert(array.count == keyPaths.count)
        self.init()
        array.enumerated().forEach { self[keyPath: keyPaths[$0]] = $1 }
    }
    
    func toArray() -> [T] {
        return Self.keyPaths.map { self[keyPath: $0] }
    }
    
}
