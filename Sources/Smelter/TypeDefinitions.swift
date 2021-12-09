public struct Shape {
    public var channels: Int
    public var width: Int
    public var height: Int
    public var depth: Int

    init(
        channels: Int,
        width: Int,
        height: Int,
        depth: Int
    ) {
        self.channels = channels
        self.width = width
        self.height = height
        self.depth = depth
    }

    static let zero = Shape(
        channels: .zero,
        width: .zero,
        height: .zero,
        depth: .zero
    )
}

extension Shape {
    func toArray() -> [Int] {
        [self.channels, self.width, self.height, self.depth]
    }

    init() { self = .zero }
}
