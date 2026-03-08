#import "MNNFaceDetector.h"
#import <MNN/Interpreter.hpp>
#import <MNN/Tensor.hpp>
#import <MNN/ImageProcess.hpp>
#import <vector>
#import <cmath>
#import <algorithm>

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

struct Prior { float cx, cy, w, h; };
struct Box   { float x1, y1, x2, y2, score; };

static std::vector<Prior> generatePriors() {
    std::vector<Prior> priors;
    for (int idx = 0; idx < (int)STRIDES.size(); idx++) {
        int stride = STRIDES[idx];
        float scaleW = (float)INPUT_W / stride;
        float scaleH = (float)INPUT_H / stride;
        int fmW = (int)std::ceil(scaleW);
        int fmH = (int)std::ceil(scaleH);
        for (int j = 0; j < fmH; j++) {
            for (int i = 0; i < fmW; i++) {
                float cx = (i + 0.5f) / scaleW;
                float cy = (j + 0.5f) / scaleH;
                for (int minBox : MIN_BOXES[idx]) {
                    priors.push_back({
                        std::clamp(cx, 0.f, 1.f),
                        std::clamp(cy, 0.f, 1.f),
                        std::clamp((float)minBox / INPUT_W, 0.f, 1.f),
                        std::clamp((float)minBox / INPUT_H, 0.f, 1.f)
                    });
                }
            }
        }
    }
    return priors;
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
            const Box &a = boxes[i], &b = boxes[j];
            float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
            float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
            float inter = std::max(ix2 - ix1, 0.f) * std::max(iy2 - iy1, 0.f);
            float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
            float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
            if (inter / (areaA + areaB - inter + 1e-5f) > IOU_THRESHOLD)
                suppressed[j] = true;
        }
    }
    return picked;
}

@implementation MNNFaceResult
@end

@interface MNNFaceDetector () {
    std::shared_ptr<MNN::Interpreter> _interpreter;
    MNN::Session *_session;
    std::vector<Prior> _priors;
}
@end

@implementation MNNFaceDetector

- (BOOL)isLoaded { return _interpreter != nullptr && _session != nullptr; }

- (instancetype)init {
    self = [super init];
    if (self) {
        NSString *path = [[NSBundle mainBundle] pathForResource:@"slim-320" ofType:@"mnn"];
        if (!path) { NSLog(@"slim-320.mnn not found"); return self; }
        _interpreter.reset(MNN::Interpreter::createFromFile(path.UTF8String));
        if (!_interpreter) { NSLog(@"Failed to create interpreter"); return self; }
        MNN::ScheduleConfig cfg;
        cfg.numThread = 2;
        _session = _interpreter->createSession(cfg);
        _priors = generatePriors();
    }
    return self;
}

- (NSArray<MNNFaceResult *> *)detectWithImage:(UIImage *)image {
    CGImageRef cgImage = image.CGImage;
    if (!cgImage || !_interpreter || !_session) return @[];

    int srcW = (int)CGImageGetWidth(cgImage);
    int srcH = (int)CGImageGetHeight(cgImage);

    // Render to RGBA
    std::vector<uint8_t> rgba(srcW * srcH * 4);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(rgba.data(), srcW, srcH, 8, srcW * 4, cs,
                                             kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGContextDrawImage(ctx, CGRectMake(0, 0, srcW, srcH), cgImage);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);

    // Preprocess: resize + normalize via MNN ImageProcess
    MNN::Tensor *inputTensor = _interpreter->getSessionInput(_session, nullptr);
    MNN::CV::ImageProcess::Config imgCfg;
    imgCfg.filterType = MNN::CV::BILINEAR;
    imgCfg.sourceFormat = MNN::CV::RGBA;
    imgCfg.destFormat = MNN::CV::RGB;
    imgCfg.normal[0] = imgCfg.normal[1] = imgCfg.normal[2] = 1.0f / 128.0f;
    imgCfg.mean[0] = imgCfg.mean[1] = imgCfg.mean[2] = 127.0f;
    std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(imgCfg));
    MNN::CV::Matrix trans;
    trans.setScale((float)srcW / INPUT_W, (float)srcH / INPUT_H);
    process->setMatrix(trans);
    _interpreter->resizeTensor(inputTensor, {1, 3, INPUT_H, INPUT_W});
    _interpreter->resizeSession(_session);
    process->convert(rgba.data(), srcW, srcH, 0, inputTensor);

    // Run
    _interpreter->runSession(_session);

    // Get outputs
    MNN::Tensor *scoresTensor = _interpreter->getSessionOutput(_session, "scores");
    MNN::Tensor *boxesTensor = _interpreter->getSessionOutput(_session, "boxes");
    MNN::Tensor scoresHost(scoresTensor, scoresTensor->getDimensionType());
    MNN::Tensor boxesHost(boxesTensor, boxesTensor->getDimensionType());
    scoresTensor->copyToHostTensor(&scoresHost);
    boxesTensor->copyToHostTensor(&boxesHost);
    const float *scores = scoresHost.host<float>();
    const float *boxesRaw = boxesHost.host<float>();

    // Decode + filter
    std::vector<Box> candidates;
    for (int i = 0; i < (int)_priors.size(); i++) {
        float conf = scores[i * 2 + 1];
        if (conf <= PROB_THRESHOLD) continue;
        const Prior &p = _priors[i];
        float cx = boxesRaw[i*4]   * CENTER_VARIANCE * p.w + p.cx;
        float cy = boxesRaw[i*4+1] * CENTER_VARIANCE * p.h + p.cy;
        float w  = std::exp(boxesRaw[i*4+2] * SIZE_VARIANCE) * p.w;
        float h  = std::exp(boxesRaw[i*4+3] * SIZE_VARIANCE) * p.h;
        candidates.push_back({cx - w/2, cy - h/2, cx + w/2, cy + h/2, conf});
    }
    if (candidates.empty()) return @[];

    // NMS + build results
    NSMutableArray<MNNFaceResult *> *results = [NSMutableArray array];
    for (const auto &b : hardNMS(candidates)) {
        MNNFaceResult *r = [[MNNFaceResult alloc] init];
        r.x = b.x1 * srcW;  r.y = b.y1 * srcH;
        r.width = (b.x2 - b.x1) * srcW;  r.height = (b.y2 - b.y1) * srcH;
        r.confidence = b.score;
        [results addObject:r];
    }
    return results;
}

@end
