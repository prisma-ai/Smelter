Pod::Spec.new do |s|
  s.name                  = 'Smelter'
  s.version               = '0.9.1'
  s.license               = { :type => 'MIT', :file => 'LICENSE' }
  s.summary               = 'Build MPSNNGraph from ONNX MetalPerformanceShadersl file'
  s.homepage              = 'https://github.com/prisma-ai/Smelter'
  s.author                = { 'Andrey Volodin' => 'a.volodin@prisma-ai.com' }
  s.source                = { :git => 'https://github.com/prisma-ai/Smelter.git', :tag => s.version.to_s }
  s.ios.deployment_target = '11.0'
  s.osx.deployment_target = '10.13'
  s.source_files          = 'Sources/**/*.{swift}', '*.py'
  s.frameworks            = 'MetalPerformanceShaders'
  s.swift_version         = "4.2"
  s.pod_target_xcconfig   = {
     'VALID_ARCHS[sdk=iphonesimulator*]' => ''
  }

  s.dependency 'SwiftProtobuf', '~> 1.1.0'
  s.dependency 'Alloy/ML', '~> 0.10.4'
end
