Pod::Spec.new do |s|
  s.name                  = 'Smelter'
  s.version               = '1.0.0'
  s.summary               = 'Build MPSNNGraph from ONNX model file'
  s.homepage              = 'https://github.com/prisma-ai/MPSNNGraphONNXBuilder'
  s.author                = { 'Andrey Volodin' => 'a.volodin@prisma-ai.com' }
  s.source                = { :git => 'https://github.com/prisma-ai/MPSNNGraphONNXBuilder.git', :tag => s.version.to_s }
  s.ios.deployment_target = '11.0'
  s.osx.deployment_target = '10.13'
  s.source_files          = 'Sources/**/*.{swift}', '*.py'
  s.frameworks            = 'MetalPerformanceShaders'
  s.swift_version         = "4.2"

  s.dependency 'SwiftProtobuf', '~> 1.1.0'
  s.dependency 'Alloy/ML', '~> 0.10.1'
end
