Pod::Spec.new do |s|
  s.name = 'Smelter'
  s.license = {
    :type => 'MIT',
    :file => 'LICENSE'
  }
  s.version = '0.10.1'

  s.summary = 'Build MPSNNGraph from ONNX file'
  s.homepage = 'https://github.com/prisma-ai/Smelter'

  s.author = {
    'Andrey Volodin' => 'a.volodin@prisma-ai.com',
    'Eugene Bokhan' => 'eugene@prisma-ai.com',
  }

  s.ios.deployment_target = '11.0'
  s.osx.deployment_target = '10.13'

  s.source = {
    :git => 'https://github.com/prisma-ai/Smelter.git',
    :tag => s.version.to_s
  }
  s.source_files = 'Sources/**/*.{swift}', '*.py'

  s.swift_version = "5.2"

  s.dependency 'SwiftProtobuf', '~> 1.7.0'
  s.dependency 'Alloy/ML', '~> 0.14.0'

  s.pod_target_xcconfig = {
    'EXCLUDED_SOURCE_FILE_NAMES' => 'ONNX2MPS.py'
  }
end
