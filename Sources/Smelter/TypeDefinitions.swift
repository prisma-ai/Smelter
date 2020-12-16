public struct Shape {
    public var channels: Int
    public var width: Int
    public var height: Int
    public var depth: Int
    
    init(channels: Int,
         width: Int,
         height: Int,
         depth: Int) {
        self.channels = channels
        self.width = width
        self.height = height
        self.depth = depth
    }
    
    static let `zero` = Shape(channels: .zero,
                              width: .zero,
                              height: .zero,
                              depth: .zero)
}

extension Shape: ArrayRepresentable {
    typealias T = Int
    
    static var keyPaths: [WritableKeyPath<Shape, Int>] {
        [\.channels, \.width, \.height, \.depth]
    }
    
    init() { self = .zero }
}


