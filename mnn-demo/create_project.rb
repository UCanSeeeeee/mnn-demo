#!/usr/bin/env ruby
require 'xcodeproj'

project_path = File.join(__dir__, 'FaceDetect.xcodeproj')
project = Xcodeproj::Project.new(project_path)

# Create target
target = project.new_target(:application, 'FaceDetect', :ios, '15.0', nil, :swift)

# Build settings
target.build_configurations.each do |config|
  config.build_settings['PRODUCT_BUNDLE_IDENTIFIER'] = 'com.demo.FaceDetect'
  config.build_settings['SWIFT_VERSION'] = '5.0'
  config.build_settings['INFOPLIST_FILE'] = 'FaceDetect/Resources/Info.plist'
  config.build_settings['ASSETCATALOG_COMPILER_APPICON_NAME'] = 'AppIcon'
  config.build_settings['SWIFT_OBJC_BRIDGING_HEADER'] = 'FaceDetect/Inference/FaceDetect-Bridging-Header.h'
  config.build_settings['CLANG_CXX_LANGUAGE_STANDARD'] = 'c++17'
  config.build_settings['CLANG_CXX_LIBRARY'] = 'libc++'
  config.build_settings['DEVELOPMENT_TEAM'] = 'W3CHGH63KV'
  config.build_settings['CODE_SIGN_IDENTITY'] = 'Apple Development'
  config.build_settings['CODE_SIGNING_REQUIRED'] = 'YES'
  config.build_settings['CODE_SIGNING_ALLOWED'] = 'YES'
  config.build_settings['CODE_SIGN_STYLE'] = 'Automatic'
  config.build_settings['TARGETED_DEVICE_FAMILY'] = '1'
  config.build_settings['LD_RUNPATH_SEARCH_PATHS'] = '$(inherited) @executable_path/Frameworks'
  config.build_settings['ALWAYS_SEARCH_USER_PATHS'] = 'NO'
  config.build_settings['GENERATE_INFOPLIST_FILE'] = 'NO'
end

# Create groups matching file structure
src_group = project.main_group.new_group('FaceDetect', 'FaceDetect')
inference_group = src_group.new_group('Inference', 'Inference')
resources_group = src_group.new_group('Resources', 'Resources')

# Swift source files
%w[AppDelegate.swift SceneDelegate.swift ViewController.swift OverlayView.swift].each do |f|
  ref = src_group.new_file(f)
  target.add_file_references([ref])
end

# ObjC++ inference bridge
inference_group.new_file('MNNFaceDetector.h')
inference_group.new_file('FaceDetect-Bridging-Header.h')
target.add_file_references([inference_group.new_file('MNNFaceDetector.mm')])

# Resources
resource_files = ['Info.plist', 'LaunchScreen.storyboard', 'slim-320.mnn']
resource_files.each do |f|
  ref = resources_group.new_file(f)
  # Info.plist should not be in build resources
  unless f == 'Info.plist'
    target.add_resources([ref])
  end
end

# Assets.xcassets
assets_ref = resources_group.new_file('Assets.xcassets')
target.add_resources([assets_ref])

project.save
puts "Created #{project_path}"
puts "Groups and files added successfully."
