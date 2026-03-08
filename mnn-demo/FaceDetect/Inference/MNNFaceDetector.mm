#import "MNNFaceDetector.h"
#import <MNN/Interpreter.hpp>
#import <MNN/MNNDefine.h>
#import <MNN/Tensor.hpp>
#import <MNN/ImageProcess.hpp>
#import <vector>
#import <cmath>
#import <algorithm>
#import <numeric>

// Constants matching detect.py
static const int INPUT_W = 320;
static const int INPUT_H = 240;
static const float CENTER_VARIANCE = 0.1f;
static const float SIZE_VARIANCE = 0.2f;
static const float PROB_THRESHOLD = 0.3f;
static const float IOU_THRESHOLD = 0.3f;
static const int CANDIDATE_SIZE = 200;

static const std::vector<std::vector<int>> MIN_BOXES = {
    {10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}
};
static const std::vector<int> STRIDES = {8, 16, 32, 64};

// ==================== Prior Box Generation ====================

struct Prior {
    float cx, cy, w, h;
};

static std::vector<Prior> generatePriors() {
    std::vector<Prior> priors;
    for (int idx = 0; idx < (int)STRIDES.size(); idx++) {
        int stride = STRIDES[idx];
        int fmW = (int)std::ceil((float)INPUT_W / stride);
        int fmH = (int)std::ceil((float)INPUT_H / stride);
        float scaleW = (float)INPUT_W / stride;
        float scaleH = (float)INPUT_H / stride;
        for (int j = 0; j < fmH; j++) {
            for (int i = 0; i < fmW; i++) {
                float cx = (i + 0.5f) / scaleW;
                float cy = (j + 0.5f) / scaleH;
                for (int minBox : MIN_BOXES[idx]) {
                    float w = (float)minBox / INPUT_W;
                    float h = (float)minBox / INPUT_H;
                    Prior p;
                    p.cx = std::min(std::max(cx, 0.0f), 1.0f);
                    p.cy = std::min(std::max(cy, 0.0f), 1.0f);
                    p.w  = std::min(std::max(w,  0.0f), 1.0f);
                    p.h  = std::min(std::max(h,  0.0f), 1.0f);
                    priors.push_back(p);
                }
            }
        }
    }
    return priors;
}

// ==================== Box Decoding ====================

struct Box {
    float x1, y1, x2, y2;
    float score;
};

static float computeIoU(const Box &a, const Box &b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(ix2 - ix1, 0.0f);
    float ih = std::max(iy2 - iy1, 0.0f);
    float inter = iw * ih;
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-5f);
}

static std::vector<Box> hardNMS(std::vector<Box> &boxes) {
    std::sort(boxes.begin(), boxes.end(), [](const Box &a, const Box &b) {
        return a.score < b.score;
    });

    int count = (int)boxes.size();
    int start = std::max(0, count - CANDIDATE_SIZE);

    std::vector<Box> picked;
    std::vector<bool> suppressed(count, false);

    for (int i = count - 1; i >= start; i--) {
        if (suppressed[i]) continue;
        picked.push_back(boxes[i]);
        for (int j = i - 1; j >= start; j--) {
            if (suppressed[j]) continue;
            if (computeIoU(boxes[i], boxes[j]) > IOU_THRESHOLD) {
                suppressed[j] = true;
            }
        }
    }
    return picked;
}

// ==================== MNNFaceResult ====================

@implementation MNNFaceResult
@end

// ==================== MNNFaceDetector ====================

@interface MNNFaceDetector () {
    std::shared_ptr<MNN::Interpreter> _interpreter;
    MNN::Session *_session;
    std::vector<Prior> _priors;
}
@end

@implementation MNNFaceDetector

- (BOOL)isLoaded {
    return _interpreter != nullptr && _session != nullptr;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"slim-320" ofType:@"mnn"];
        if (!modelPath) {
            NSLog(@"ERROR: slim-320.mnn not found in bundle");
            return self;
        }
        _interpreter.reset(MNN::Interpreter::createFromFile(modelPath.UTF8String));
        if (!_interpreter) {
            NSLog(@"ERROR: Failed to create MNN interpreter");
            return self;
        }
        MNN::ScheduleConfig config;
        config.numThread = 2;
        _session = _interpreter->createSession(config);
        _priors = generatePriors();
        NSLog(@"MNNFaceDetector initialized, priors: %lu", (unsigned long)_priors.size());
    }
    return self;
}

- (NSArray<MNNFaceResult *> *)detectWithImage:(UIImage *)image {
    CGImageRef cgImage = image.CGImage;
    if (!cgImage) return @[];

    int srcW = (int)CGImageGetWidth(cgImage);
    int srcH = (int)CGImageGetHeight(cgImage);

    // Render CGImage to RGBA buffer
    std::vector<uint8_t> rgba(srcW * srcH * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgba.data(), srcW, srcH, 8, srcW * 4, colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
    );
    CGContextDrawImage(ctx, CGRectMake(0, 0, srcW, srcH), cgImage);
    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);

    return [self runInference:rgba.data() width:srcW height:srcH channels:4 format:MNN::CV::RGBA];
}

- (NSArray<MNNFaceResult *> *)detectWithPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    int srcW = (int)CVPixelBufferGetWidth(pixelBuffer);
    int srcH = (int)CVPixelBufferGetHeight(pixelBuffer);
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);

    NSArray<MNNFaceResult *> *results = [self runInference:baseAddress width:srcW height:srcH channels:4 format:MNN::CV::BGRA];

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return results;
}

- (NSArray<MNNFaceResult *> *)runInference:(const uint8_t *)data
                                     width:(int)srcW
                                    height:(int)srcH
                                  channels:(int)channels
                                    format:(MNN::CV::ImageFormat)format {
    if (!_interpreter || !_session) return @[];

    MNN::Tensor *inputTensor = _interpreter->getSessionInput(_session, nullptr);

    // Use MNN::CV::ImageProcess for preprocessing
    MNN::CV::ImageProcess::Config imgConfig;
    imgConfig.filterType = MNN::CV::BILINEAR;
    imgConfig.sourceFormat = format;
    imgConfig.destFormat = MNN::CV::RGB;
    // (pixel - 127) / 128 = pixel * (1/128) + (-127/128)
    imgConfig.normal[0] = 1.0f / 128.0f;
    imgConfig.normal[1] = 1.0f / 128.0f;
    imgConfig.normal[2] = 1.0f / 128.0f;
    imgConfig.mean[0] = 127.0f;
    imgConfig.mean[1] = 127.0f;
    imgConfig.mean[2] = 127.0f;

    std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(imgConfig));

    MNN::CV::Matrix trans;
    trans.setScale((float)srcW / INPUT_W, (float)srcH / INPUT_H);
    process->setMatrix(trans);

    _interpreter->resizeTensor(inputTensor, {1, 3, INPUT_H, INPUT_W});
    _interpreter->resizeSession(_session);

    process->convert(data, srcW, srcH, 0, inputTensor);

    _interpreter->runSession(_session);

    // Get outputs
    MNN::Tensor *scoresTensor = _interpreter->getSessionOutput(_session, "scores");
    MNN::Tensor *boxesTensor = _interpreter->getSessionOutput(_session, "boxes");

    // Copy to host
    MNN::Tensor scoresHost(scoresTensor, scoresTensor->getDimensionType());
    scoresTensor->copyToHostTensor(&scoresHost);
    MNN::Tensor boxesHost(boxesTensor, boxesTensor->getDimensionType());
    boxesTensor->copyToHostTensor(&boxesHost);

    const float *scores = scoresHost.host<float>();
    const float *boxesRaw = boxesHost.host<float>();
    int numPriors = (int)_priors.size();

    // Decode boxes and filter by confidence
    std::vector<Box> candidates;
    for (int i = 0; i < numPriors; i++) {
        float conf = scores[i * 2 + 1]; // face class score
        if (conf <= PROB_THRESHOLD) continue;

        const Prior &p = _priors[i];
        float dx = boxesRaw[i * 4 + 0];
        float dy = boxesRaw[i * 4 + 1];
        float dw = boxesRaw[i * 4 + 2];
        float dh = boxesRaw[i * 4 + 3];

        // Decode center
        float cx = dx * CENTER_VARIANCE * p.w + p.cx;
        float cy = dy * CENTER_VARIANCE * p.h + p.cy;
        float w  = std::exp(dw * SIZE_VARIANCE) * p.w;
        float h  = std::exp(dh * SIZE_VARIANCE) * p.h;

        // Convert to corner form
        Box box;
        box.x1 = cx - w / 2.0f;
        box.y1 = cy - h / 2.0f;
        box.x2 = cx + w / 2.0f;
        box.y2 = cy + h / 2.0f;
        box.score = conf;
        candidates.push_back(box);
    }

    if (candidates.empty()) return @[];

    // NMS
    std::vector<Box> picked = hardNMS(candidates);

    // Scale to original image size and return results
    NSMutableArray<MNNFaceResult *> *results = [NSMutableArray array];
    for (const auto &b : picked) {
        MNNFaceResult *r = [[MNNFaceResult alloc] init];
        r.x = b.x1 * srcW;
        r.y = b.y1 * srcH;
        r.width  = (b.x2 - b.x1) * srcW;
        r.height = (b.y2 - b.y1) * srcH;
        r.confidence = b.score;
        [results addObject:r];
    }
    return results;
}

@end
