#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreVideo/CoreVideo.h>

NS_ASSUME_NONNULL_BEGIN

@interface MNNFaceResult : NSObject
@property (nonatomic) float x;
@property (nonatomic) float y;
@property (nonatomic) float width;
@property (nonatomic) float height;
@property (nonatomic) float confidence;
@end

@interface MNNFaceDetector : NSObject
@property (nonatomic, readonly) BOOL isLoaded;
- (instancetype)init;
- (NSArray<MNNFaceResult *> *)detectWithImage:(UIImage *)image;
- (NSArray<MNNFaceResult *> *)detectWithPixelBuffer:(CVPixelBufferRef)pixelBuffer;
@end

NS_ASSUME_NONNULL_END
