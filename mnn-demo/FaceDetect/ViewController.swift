import UIKit

class ViewController: UIViewController {

    private let detector = MNNFaceDetector()
    private let imageView = UIImageView()
    private let overlayView = OverlayView()
    private let pickButton = UIButton(type: .system)
    private let statusLabel = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        setupUI()
    }

    private func setupUI() {
        // Result image view - top area
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = UIColor(white: 0.95, alpha: 1)
        imageView.clipsToBounds = true
        imageView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(imageView)

        // Overlay for bounding boxes
        overlayView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(overlayView)

        // Status label
        statusLabel.font = .systemFont(ofSize: 13)
        statusLabel.textAlignment = .center
        statusLabel.textColor = .darkGray
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(statusLabel)

        // Pick button
        pickButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .medium)
        pickButton.translatesAutoresizingMaskIntoConstraints = false
        pickButton.addTarget(self, action: #selector(pickPhoto), for: .touchUpInside)
        view.addSubview(pickButton)

        NSLayoutConstraint.activate([
            imageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),

            overlayView.topAnchor.constraint(equalTo: imageView.topAnchor),
            overlayView.leadingAnchor.constraint(equalTo: imageView.leadingAnchor),
            overlayView.trailingAnchor.constraint(equalTo: imageView.trailingAnchor),
            overlayView.bottomAnchor.constraint(equalTo: imageView.bottomAnchor),

            statusLabel.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: 12),
            statusLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            statusLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),

            pickButton.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 16),
            pickButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            pickButton.widthAnchor.constraint(equalToConstant: 200),
            pickButton.heightAnchor.constraint(equalToConstant: 50),
            pickButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),

            imageView.bottomAnchor.constraint(equalTo: statusLabel.topAnchor, constant: -12),
        ])

        // Model load status on button
        if detector.isLoaded {
            pickButton.setTitle("选择图片（模型已加载）", for: .normal)
        } else {
            pickButton.setTitle("选择图片（模型加载失败）", for: .normal)
            pickButton.isEnabled = false
        }
    }

    @objc private func pickPhoto() {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = self
        present(picker, animated: true)
    }

    private func detectOnImage(_ image: UIImage) {
        imageView.image = image
        overlayView.faces = []
        statusLabel.text = "检测中..."

        guard let cgImage = image.cgImage else { return }
        let imgSize = CGSize(width: cgImage.width, height: cgImage.height)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let results = self.detector.detect(with: image)
            DispatchQueue.main.async {
                self.showResults(results, imageSize: imgSize)
            }
        }
    }

    private func showResults(_ results: [MNNFaceResult], imageSize: CGSize) {
        let viewRect = imageView.bounds
        guard viewRect.width > 0, viewRect.height > 0 else { return }

        // Aspect-fit transform from image coords to imageView coords
        let scaleX = viewRect.width / imageSize.width
        let scaleY = viewRect.height / imageSize.height
        let scale = min(scaleX, scaleY)
        let offsetX = (viewRect.width - imageSize.width * scale) / 2
        let offsetY = (viewRect.height - imageSize.height * scale) / 2

        var faces: [FaceBox] = []
        for r in results {
            let rect = CGRect(
                x: CGFloat(r.x) * scale + offsetX,
                y: CGFloat(r.y) * scale + offsetY,
                width: CGFloat(r.width) * scale,
                height: CGFloat(r.height) * scale
            )
            faces.append(FaceBox(rect: rect, confidence: r.confidence))
        }
        overlayView.faces = faces
        statusLabel.text = "检测到 \(faces.count) 张人脸"
    }
}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        picker.dismiss(animated: true)
        if let image = info[.originalImage] as? UIImage {
            detectOnImage(image)
        }
    }
}
