// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "Smelter",
    platforms: [
        .iOS(.v11),
        .macOS(.v10_13)
    ],
    products: [
        .library(name: "Smelter",
                 targets: ["Smelter"]),
    ],
    dependencies: [
        .package(url: "https://github.com/s1ddok/Alloy.git",
                 .upToNextMajor(from: "0.16.6")),
        .package(name: "SwiftProtobuf",
                 url: "https://github.com/apple/swift-protobuf.git",
                 .exact("1.7.0"))
    ],
    targets: [
        .target(name: "Smelter",
                dependencies: ["Alloy", "SwiftProtobuf"])
    ]
)
