// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "Smelter",
    platforms: [
        .iOS(.v11),
        .macOS(.v10_13),
    ],
    products: [
        .library(
            name: "Smelter",
            targets: ["Smelter"]
        ),
    ],
    dependencies: [
        .package(
            name: "SwiftProtobuf",
            url: "https://github.com/apple/swift-protobuf.git",
            .upToNextMajor(from: "1.18.0")
        ),
    ],
    targets: [
        .target(
            name: "Smelter",
            dependencies: ["SwiftProtobuf"]
        ),
    ]
)
