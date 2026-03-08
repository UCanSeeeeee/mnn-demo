#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

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
- (NSArray<MNNFaceResult *> *)detectWithImage:(UIImage *)image;
@end

NS_ASSUME_NONNULL_END
