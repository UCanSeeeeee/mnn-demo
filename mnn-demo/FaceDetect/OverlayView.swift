import UIKit

struct FaceBox {
    let rect: CGRect
    let confidence: Float
}

class OverlayView: UIView {
    var faces: [FaceBox] = [] {
        didSet { setNeedsDisplay() }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isUserInteractionEnabled = false
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.setStrokeColor(UIColor.green.cgColor)
        ctx.setLineWidth(2.0)

        let attrs: [NSAttributedString.Key: Any] = [
            .foregroundColor: UIColor.green,
            .font: UIFont.systemFont(ofSize: 14, weight: .bold)
        ]

        for face in faces {
            ctx.stroke(face.rect)
            let text = String(format: "%.2f", face.confidence)
            let point = CGPoint(x: face.rect.origin.x, y: face.rect.origin.y - 18)
            (text as NSString).draw(at: point, withAttributes: attrs)
        }
    }
}
